#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime
import os
import time
import shutil
import sys
import socket

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute, visualize
from dask.distributed import Client, LocalCluster, wait
from dask_jobqueue import SLURMCluster
import ztfquery.io
import numpy as np


import list_format


ztf_filtercodes = ['zg', 'zr', 'zi', 'all']
poloka_func = []


def run_and_log(cmd, logger=None):
    if logger:
        logger.info("Running command: \"{}\"".format(" ".join([str(s) for s in cmd])))
        start_time = time.perf_counter()

    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)

    if logger:
        logger.info("Done running command. Elapsed time={}".format(time.perf_counter() - start_time))
        logger.info("Command stdout/stderr output:")
        logger.info(out.stdout)
        logger.info("=========================== output end ===========================")

    return out.returncode


def make_catalog(folder, logger):
    if args.retrieve_calibrated:
        sciimg_path = ztfquery.io.get_file(folder.name + "_sciimg.fits", downloadit=False)
        shutil.copy2(sciimg_path, folder.joinpath("calibrated.fits"))

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


def smphot(results, cwd, ztfname, filtercode):
    logger = logging.getLogger("{}-{}".format(ztfname, filtercode))
    logger.addHandler(logging.FileHandler(args.wd.joinpath("{}/{}/output.log".format(ztfname, filtercode)), mode='a'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())
    logger.info("Running reduction {}".format(args.func))

    quadrant_root = cwd.joinpath("{}/{}".format(ztfname, filtercode))
    quadrant_folders = [folder for folder in quadrant_root.glob("*".format(ztfname, filtercode)) if folder.is_dir()]
    quadrant_folders = list(filter(lambda x: x.joinpath("psfstars.list").exists(), quadrant_folders))

    logger.info("Determining best seeing quadrant...")
    seeing = {}
    for folder in quadrant_folders:
        calibrated_file = folder.joinpath("calibrated.fits")
        with fits.open(calibrated_file) as hdul:
            seeing[folder] = hdul[0].header['seseeing']

    seeing_df = pd.DataFrame.from_dict(seeing, orient='index')

    idxmin = seeing_df.idxmin().values[0]
    minseeing = seeing_df.at[idxmin, 0]

    logger.info("Best seeing quadrant: {}". format(idxmin))
    logger.info("  with seeing={}".format(minseeing))

    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    with fits.open(pathlib.Path(idxmin).joinpath("calibrated.fits")) as hdul:
        w = WCS(hdul[0].header)

    ra_px, dec_px = w.world_to_pixel(SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg'))

    driver_path = quadrant_root.joinpath("{}_driver_{}".format(ztfname, filtercode))
    logger.info("Writing driver file at location {}".format(driver_path))
    with open(driver_path, 'w') as f:
        f.write("OBJECTS\n")
        f.write("{} {} DATE_MIN={} DATE_MAX={} NAME={} TYPE=0 BAND={}\n".format(ra_px[0],
                                                                                dec_px[0],
                                                                                sn_parameters['t_inf'].values[0],
                                                                                sn_parameters['t_sup'].values[0],
                                                                                ztfname,
                                                                                filtercode))
        f.write("IMAGES\n")
        for quadrant_folder in seeing_df.index:
            f.write("{}\n".format(quadrant_folder))
        f.write("PHOREF\n")
        f.write("{}\n".format(idxmin))
        f.write("PMLIST\n")



    logger.info("Building Gaia catalog...")
    # Create GAIA catalog
    gaia_cat = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_cat.reset_index(drop=True, inplace=True)

    gaia_cat = gaia_cat.assign(ra_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)
    gaia_cat = gaia_cat.assign(dec_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)

    gaia_cat = gaia_cat.rename(columns={'pmde': 'pmdec', 'plx': 'parallax', 'e_pmra': 'pmra_error', 'e_pmde': 'pmdec_error', 'gmag': 'g', 'bpmag': 'bp', 'rpmag': 'rp', 'e_gmag': 'g_error', 'e_bpmag': 'bperror', 'e_bpmag': 'bp_error', 'e_rpmag': 'rp_error'})

    gaia_cat = gaia_cat[['ra', 'dec', 'ra_error', 'dec_error', 'pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'g', 'bp', 'rp', 'g_error', 'bp_error', 'rp_error']]

    gaia_path = args.wd.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    np.save(gaia_path, gaia_cat.to_records(index=False))


    with open(driver_path, 'a') as f:
        f.write(str(quadrant_root.joinpath("pmfit/pmcatalog.list")))

    run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir=pmfit"], logger=logger)


    run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir=pmfit", "--plot-dir=pmfit_plot"], logger=logger)

    return True


poloka_func.append({'reduce': smphot})


poloka_func = dict(zip([list(func.values())[0].__name__ for func in poloka_func], poloka_func))


def launch(quadrant, wd, ztfname, filtercode, func, scratch=None):
    quadrant_dir = wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))

    logger = None
    if func != 'clean':
        logger = logging.getLogger(quadrant)
        logger.addHandler(logging.FileHandler(quadrant_dir.joinpath("output.log"), mode='a'))
        logger.setLevel(logging.INFO)
        logger.info(datetime.datetime.today())
        logger.info("Current directory: {}".format(quadrant_dir))
        logger.info("Running {}".format(func))

        logger.info("Quadrant directory: {}".format(quadrant_dir))

        if scratch:
            logger.info("Using scratch space {}".format(scratch))
            logger.info(" Parent exists={}".format(scratch.parent.exists()))
            quadrant_scratch = scratch.joinpath(quadrant)
            quadrant_scratch.mkdir(exist_ok=True, parents=True)
            logger.info("Successfully created quadrant working dir in scratch space")
            files = list(quadrant_dir.glob("*"))

            [shutil.copyfile(f, quadrant_scratch.joinpath(f.name)) for f in files]
            quadrant_dir = quadrant_scratch
            logger.info("Successfully copyed files from sps to scratchspace")

    result = False
    try:
        result = poloka_func[func]['map'](quadrant_dir, logger)
    except Exception as e:
        print("")
        print("In folder {}".format(quadrant_dir))
        print(e)
    finally:
        if scratch and func != 'clean':
            files = list(quadrant_dir.glob("*"))
            [shutil.copy2(f, wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))) for f in files]
            [f.unlink() for f in files]
            quadrant_dir.rmdir()

    if func != 'clean':
        logger.info("Done.")

    if scratch and func != 'clean':
        files = list(quadrant_dir.glob("*"))
        [shutil.copyfile(f, wd.joinpath("{}/{}/{}/{}".format(ztfname, filtercode, quadrant, f.name))) for f in files]
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
    argparser.add_argument('--retrieve-calibrated', dest='retrieve_calibrated', action='store_true')
    argparser.add_argument('--cosmodr', type=pathlib.Path, help="Cosmo DR folder.")
    argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path)

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()


    if args.cosmodr:
        cosmo_dr_folder = args.cosmodr
    elif 'COSMO_DR_FOLDER' in os.environ.keys():
        cosmo_dr_folder = pathlib.Path(os.environ.get("COSMO_DR_FOLDER"))
    else:
        print("Cosmo DR folder not set! Either set COSMO_DR_FOLDER environnement variable or use the --cosmodr parameter.")
        exit(-1)

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


    if args.scratch:
        args.scratch.mkdir(exist_ok=True, parents=True)

        import signal
        import atexit
        def delete_scratch_at_exit(scratch_dir):
            shutil.rmtree(scratch_dir)

        atexit.register(delete_scratch_at_exit, scratch_dir=args.scratch)


    if args.cluster:
        cluster = SLURMCluster(cores=24,
                               processes=24,
                               memory="64GB",
                               project="ztf",
                               walltime="12:00:00",
                               queue="htc",
                               job_extra=["-L sps"])
        cluster.scale(jobs=10)
        client = Client(cluster)
        print(client.dashboard_link, flush=True)
        print(socket.gethostname(), flush=True)
        client.wait_for_workers(10)
    else:
        if args.n_jobs == 1:
            dask.config.set(scheduler='synchronous')

        localCluster = LocalCluster(n_workers=args.n_jobs, dashboard_address='localhost:8787', threads_per_worker=1, nanny=False)
        client = Client(localCluster)
        print("Dask dashboard at: {}".format(client.dashboard_link))


    jobs = []
    quadrant_count = 0
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            print("Building job list for {}-{}... ".format(ztfname, filtercode), end="", flush=True)
            results = None

            if 'map' in poloka_func[args.func].keys():
                quadrants = list(map(lambda x: x.stem, filter(lambda x: x.is_dir(), args.wd.joinpath("{}/{}".format(ztfname, filtercode)).glob("*"))))
                quadrant_count += len(quadrants)

                results = [delayed(launch)(quadrant, args.wd, ztfname, filtercode, args.func, args.scratch) for quadrant in quadrants]
                print("Found {} quadrants.".format(len(quadrants)))

            if 'reduce' in poloka_func[args.func].keys():
                results = [delayed(poloka_func[args.func]['reduce'])(results, args.wd, ztfname, filtercode)]

            jobs.extend(results)

    print("")
    #visualize(jobs, filename="/home/llacroix/mygraph.png")
    print("Running")
    start_time = time.perf_counter()
    fjobs = client.compute(jobs)
    wait(fjobs)
    print("Done. Elapsed time={}".format(time.perf_counter() - start_time))

    client.close()
