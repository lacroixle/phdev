#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime
import os
import time
import sys

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import dask
from dask import delayed, compute, visualize
from dask.distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster

import list_format


ztf_filtercodes = ['zg', 'zr', 'zi', 'all']
poloka_func = []


def run_and_log(cmd, logger=None):
    out = subprocess.run(cmd, capture_output=True)
    if logger:
        logger.info(out.stdout.decode('utf-8'))
        logger.error(out.stderr.decode('utf-8'))

    return out.returncode

poloka_func.append({'map': run_and_log})


def make_catalog(folder, logger):
    run_and_log(["make_catalog", folder], logger)
    return folder.joinpath("se.list").exists()

poloka_func.append({'map': make_catalog})


def mkcat2(folder, logger):
    run_and_log(["mkcat2", folder], logger)
    return folder.joinpath("standalone_stars.list").exists()

poloka_func.append({'map': mkcat2})


def makepsf(folder, logger):
    run_and_log(["makepsf", folder], logger)
    return folder.joinpath("psfstars.list").exists()

poloka_func.append({'map': makepsf})


def pipeline(folder, logger):
    if not make_catalog(folder, logger):
        return False

    if not mkcat2(folder, logger):
        return False

    if not makepsf(folder, logger):
        return False

    return True

poloka_func.append({'map': pipeline})


files_to_keep = ["elixir.fits", "mask.fits", "calibrated.fits", ".dbstuff"]
def clean(folder, logger):
    # calibrated.fits header gets modified, it might be problematic at some point (or not)
    if args.dry_run:
        print("In quadrant folder {}".format(folder))

    files = list(folder.glob("*"))
    files_to_delete = [file_to_delete for file_to_delete in files if file_to_delete.name not in files_to_keep]

    for file_to_delete in files_to_delete:
        if not args.dry_run:
            file_to_delete.unlink()
        else:
            print("File to delete: {}".format(file_to_delete))

    return True

poloka_func.append({'map': clean})


# Extract data from standalone stars and plot several distributions
def stats(folder, logger):
    def _extract_from_list(list_filename, hdfstore):
        list_path = folder.joinpath(list_filename).with_suffix(".list")

        if not list_path.exists():
            return False

        with open(list_path, mode='r') as f:
            global_params, df = list_format.read_list(f)

        hdfstore.put(list_path.stem, df)
        hdfstore.put("{}_globals".format(list_path.stem), pd.DataFrame([global_params]))

        return True

    import warnings
    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

    with pd.HDFStore(folder.joinpath("lists.hdf5"), mode='w') as hdfstore:
        # From make_catalog
        _extract_from_list("se", hdfstore)

        # From mkcat2
        cont = _extract_from_list("standalone_stars", hdfstore)

        if not cont:
            return True

        _extract_from_list("aperse", hdfstore)

        # From calibrated.fits
        keywords = ['sexsky', 'sexsigma', 'bscale', 'bzero', 'origsatu', 'saturlev', 'backlev', 'back_sub', 'seseeing', 'gfseeing']

        calibrated = {}
        with fits.open(folder.joinpath("calibrated.fits")) as hdul:
            for keyword in keywords:
                calibrated[keyword] = hdul[0].header[keyword]

            hdfstore.put('calibrated', pd.DataFrame([calibrated]))

        # From makepsf
        cont = _extract_from_list("psfstars", hdfstore)

        if not cont:
            return True

        _extract_from_list("psftuples", hdfstore)

    return True


def stats_reduce(results, cwd, ztfname, filtercode):
    # Seeing histogram
    folders = [folder for folder in cwd.glob("*") if folder.is_dir()]

    seseeings = []
    for folder in folders:
        hdfstore_path = folder.joinpath("lists.hdf5")

        if hdfstore_path.exists():
            with pd.HDFStore(hdfstore_path, mode='r') as hdfstore:
                if '/calibrated' in hdfstore.keys():
                    calibrated_df = hdfstore.get('/calibrated')
                    seseeings.append(float(calibrated_df['seseeing']))

    # plt.hist(seseeings, bins=60, range=[0.5, 3], color='xkcd:dark grey', histtype='step')
    # plt.grid()
    # plt.savefig(cwd.joinpath("{}-{}_seseeing_dist.png".format(ztfname, filtercode)), dpi=300)
    # plt.close()

    with open(cwd.joinpath("{}-{}_failures.txt".format(ztfname, filtercode)), 'w') as f:
        # Failure rates
        def _failure_rate(listname, func):
            success_count = 0
            for folder in folders:
                if folder.joinpath("{}.list".format(listname)).exists():
                    success_count += 1

            f.writelines(["For {}:\n".format(func),
                          " Success={}/{}\n".format(success_count, len(folders)),
                          " Rate={}\n\n".format(float(success_count)/len(folders))])

        _failure_rate("se", 'make_catalog')
        _failure_rate("standalone_stars", 'mkcat2')
        _failure_rate("psfstars", 'makepsf')

    return sum(results)/len(results)

poloka_func.append({'map': stats, 'reduce': stats_reduce})


poloka_func = dict(zip([func['map'].__name__ for func in poloka_func], poloka_func))


def launch(quadrant, wd, ztfname, filtercode, func, scratch=None):
    quadrant_dir = wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))

    if scratch:
        quadrant_scratch = scratch.joinpath(quadrant)
        quadrant_scratch.mkdir(exist_ok=True)
        files = list(quadrant_dir.glob("*"))

        [shuti.copy2(f, quadrant_scratch) for f in files]
        quadrant_dir = quadrant_scratch

    logger = None
    if func != 'clean':
        logger = logging.getLogger(quadrant)
        logger.addHandler(logging.FileHandler(quadrant_dir.joinpath("output.log"), mode='a'))
        logger.setLevel(logging.INFO)
        logger.info(datetime.datetime.today())
        logger.info("Current directory: {}".format(quadrant_dir))
        logger.info("Running {}".format(func))

    result = False
    try:
        result = poloka_func[func]['map'](quadrant_dir, logger)
    except Exception as e:
        print("")
        print("In folder {}".format(curdir))
        print(e)

    # if not args.dry_run:
    #     if result:
    #         print(".", end="", flush=True)
    #     else:
    #         print("x", end="", flush=True)
    #

    if func != 'clean':
        logger.info("Done.")

    if scratch:
        files = list(quadrant_dir.glob("*"))
        [shutil.copy2(f, wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant)))]
        [f.unlink() for f in files]
        quadrant_dir.rmdir()

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, help="If provided, perform computation on one SN1a. If it points to a valid text file, will perform computation on all keys. If not provided, process the whole working directory.")
    argparser.add_argument('-j', dest='n_jobs', type=int, default=1)
    argparser.add_argument('--wd', type=pathlib.Path, help="Working directory")
    argparser.add_argument('--filtercode', choices=ztf_filtercodes, default='all', help="Only perform computations on one or all filters.")
    argparser.add_argument('--func', type=str, choices=poloka_func.keys(), default='pipeline')
    argparser.add_argument('--dry-run', dest='dry_run', action='store_true')
    argparser.add_argument('--no-map', dest='no_map', action='store_true')
    argparser.add_argument('--no-reduce', dest='no_reduce', action='store_true')
    argparser.add_argument('--cluster', action='store_true')
    argparser.add_argument('--scratch', type=pathlib.Path)

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()

    filtercodes = ztf_filtercodes[:3]
    if args.filtercode != 'all':
        filtercodes = [args.filtercode]


    ztfnames = None
    if args.ztfname is not None:
        if args.ztfname.stem == str(args.ztfname):
            ztfnames = [str(args.ztfname)]
        else:
            args.ztfname = args.ztfname.expanduser().resolve()
            if args.ztfname.exists():
                with open(args.ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
            else:
                pass


    if args.cluster:
        cluster = SLURMCluster(cores=4,
                               processes=4,
                               memory="8GB",
                               project="ztf",
                               walltime="12:00:00",
                               queue="htc",
                               job_extra=["-L sps"])
        cluster.scale(jobs=100)
        client = Client(cluster)
        print(client.dashboard_link)
        #client.wait_for_workers(3)
    else:
        localCluster = LocalCluster(n_workers=args.n_jobs, dashboard_address='localhost:8787')
        client = Client(localCluster)
        print("Local cluster!")
        print(client.dashboard_link)

    if args.n_jobs == 1:
        dask.config.set(scheduler='synchronous')

    jobs = []
    quadrant_count = 0
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            print("Building job list for {}-{}... ".format(ztfname, filtercode), end="", flush=True)
            quadrants = list(map(lambda x: x.stem, filter(lambda x: x.is_dir(), args.wd.joinpath("{}/{}".format(ztfname, filtercode)).glob("*"))))
            quadrant_count += len(quadrants)

            results = [delayed(launch)(quadrant, args.wd, ztfname, filtercode, args.func) for quadrant in quadrants]

            if 'reduce' in poloka_func[args.func].keys():
                results = [delayed(poloka_func[args.func]['reduce'])(results, args.wd, ztfname, filtercode)]

            print("Found {} quadrants.".format(len(quadrants)))
            jobs.extend(results)


    #visualize(jobs, filename="/home/llacroix/mygraph.png")
    print("Running")
    start_time = time.perf_counter()
    fjobs = client.compute(jobs)
    wait(fjobs)
    print("Done. Elapsed time={}".format(time.perf_counter() - start_time))

    client.close()
