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
import copy
import traceback

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, wait, get_worker
from dask_jobqueue import SLURMCluster, SGECluster
import ztfquery.io
import numpy as np
from skimage.morphology import label


import utils

matplotlib.use('Agg')

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


def make_catalog(quadrant_folder, logger):
    logger.info("Retrieving calibrated.fits...")
    sciimg_path = ztfquery.io.get_file(quadrant_folder.name + "_sciimg.fits", downloadit=False)
    logger.info("Located at {}".format(sciimg_path))
    shutil.copyfile(sciimg_path, quadrant_folder.joinpath("calibrated.fits"))

    run_and_log(["make_catalog", quadrant_folder, "-O", "-S"], logger)

    # Compute number of masked cosmics
    with fits.open(quadrant_folder.joinpath("cosmic.fits.gz")) as hdul:
        _, cosmic_count = label(hdul[0].data, return_num=True)

    aperse_listtable = utils.ListTable.from_filename(quadrant_folder.joinpath("se.list"))
    aperse_listtable.header['cosmic_count'] = cosmic_count
    aperse_listtable.write()

    if cosmic_count > 500:
        quadrant_folder.joinpath("se.list").unlink(missing_ok=True)

    return quadrant_folder.joinpath("se.list").exists()


poloka_func.append({'map': make_catalog})


def mkcat2(folder, logger):
    run_and_log(["mkcat2", folder, "-o"], logger)
    return folder.joinpath("standalone_stars.list").exists()


poloka_func.append({'map': mkcat2})


def makepsf(folder, logger):
    run_and_log(["makepsf", folder, "-f"], logger)
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


#files_to_keep = ["elixir.fits", "mask.fits", "deads.fits.gz", ".dbstuff"]
files_to_keep = ["elixir.fits", "deads.fits.gz", ".dbstuff"]
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


def clean_reduce(folder, ztfname, filtercode, logger):
    # Delete all files
    files_to_delete = list(filter(lambda f: f.is_file(), list(folder.glob("*"))))
    [f.unlink() for f in files_to_delete]

    shutil.rmtree(folder.joinpath("pmfit"))
    shutil.rmtree(folder.joinpath("pmfit_plot"))


poloka_func.append({'map': clean, 'reduce': clean_reduce})


# Extract data from standalone stars and plot several distributions
def stats(folder, logger):
    def _extract_from_list(list_filename, hdfstore):
        list_path = folder.joinpath(list_filename).with_suffix(".list")

        if not list_path.exists():
            return False

        with open(list_path, mode='r') as f:
            global_params, df, _, _ = utils.read_list(f)

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


def stats_reduce(cwd, ztfname, filtercode, logger):
    # Seeing histogram
    folders = [folder for folder in cwd.glob("*") if folder.is_dir()]

    logger.info("Plotting fitted seeing histogram")
    seseeings = []
    for folder in folders:
        hdfstore_path = folder.joinpath("lists.hdf5")

        if hdfstore_path.exists():
            with pd.HDFStore(hdfstore_path, mode='r') as hdfstore:
                if '/calibrated' in hdfstore.keys():
                    calibrated_df = hdfstore.get('/calibrated')
                    seseeings.append(float(calibrated_df['seseeing']))

    plt.hist(seseeings, bins=int(len(seseeings)/4), range=[0.5, 3], color='xkcd:dark grey', histtype='step')
    plt.grid()
    plt.xlabel("Seeing")
    plt.ylabel("Count")
    plt.savefig(cwd.joinpath("{}-{}_seseeing_dist.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

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

    logger.info("Plotting computing time histograms")
    # Plot results_*.csv histogram
    results = list(cwd.glob("results_*.csv"))
    for result in results:
        func = str(result.stem).split("_")[1]
        result_df = pd.read_csv(result)
        computation_times = (result_df['time_end'] - result_df['time_start']).to_numpy()
        plt.hist(computation_times, bins=int(len(result_df)/4), histtype='step')
        plt.xlabel("Computation time (s)")
        plt.ylabel("Count")
        plt.title("Computation time for {}".format(result))
        plt.grid()
        plt.savefig(cwd.joinpath("{}-{}_{}_compute_time_dist.png".format(ztfname, filtercode, func)), dpi=300)
        plt.close()


poloka_func.append({'map': stats, 'reduce': stats_reduce})


def smphot(cwd, ztfname, filtercode, logger):
    quadrant_folders = [folder for folder in cwd.glob("ztf_*".format(ztfname, filtercode)) if folder.is_dir()]
    quadrant_folders = list(filter(lambda x: x.joinpath("psfstars.list").exists(), quadrant_folders))

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    seeings = {}

    for quadrant in quadrant_folders:
        calibrated_file = quadrant.joinpath("calibrated.fits")
        with fits.open(calibrated_file) as hdul:
            seeings[quadrant] = (hdul[0].header['seseeing'], hdul[0].header['fieldid'])

    fieldids = list(set([seeing[1] for seeing in seeings.values()]))
    fieldids_count = [sum([1 for f in seeings.values() if f[1]==fieldid]) for fieldid in fieldids]
    maxcount_field = fieldids[np.argmax(fieldids_count)]

    logger.info("{} different field ids".format(len(fieldids)))
    logger.info(fieldids)
    logger.info(fieldids_count)
    logger.info("Max quadrant field={}".format(maxcount_field))

    #seeing_df = pd.DataFrame.from_dict([{quadrant: seeings[quadrant][0]} for quadrant in seeings.keys() if seeings[quadrant][1]==maxcount_field], orient='index')
    seeing_df = pd.DataFrame([[quadrant, seeings[quadrant][0]] for quadrant in seeings.keys() if seeings[quadrant][1]==maxcount_field], columns=['quadrant', 'seeing'])
    seeing_df = seeing_df.set_index(['quadrant'])

    idxmin = seeing_df.idxmin().values[0]
    minseeing = seeing_df.at[idxmin, 'seeing']

    logger.info("Best seeing quadrant: {}". format(idxmin))
    logger.info("  with seeing={}".format(minseeing))

    logger.info("Reading SN1a parameters")
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    logger.info("Reading reference WCS")
    with fits.open(pathlib.Path(idxmin).joinpath("calibrated.fits")) as hdul:
        w = WCS(hdul[0].header)

    ra_px, dec_px = w.world_to_pixel(SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg'))

    logger.info("Writing driver file")
    driver_path = cwd.joinpath("{}_driver_{}".format(ztfname, filtercode))
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
        for quadrant_folder in quadrant_folders:
            f.write("{}\n".format(quadrant_folder))
        f.write("PHOREF\n")
        f.write("{}\n".format(idxmin))
        f.write("PMLIST\n")
        f.write(str(cwd.joinpath("pmfit/pmcatalog.list")))

    # Create GAIA catalog
    logger.info("Building Gaia catalog")

    gaia_cat = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_cat.reset_index(drop=True, inplace=True)

    gaia_cat = gaia_cat.assign(ra_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)
    gaia_cat = gaia_cat.assign(dec_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)

    gaia_cat = gaia_cat.rename(columns={'pmde': 'pmdec', 'plx': 'parallax', 'e_pmra': 'pmra_error', 'e_pmde': 'pmdec_error', 'gmag': 'g', 'bpmag': 'bp', 'rpmag': 'rp', 'e_gmag': 'g_error', 'e_bpmag': 'bperror', 'e_bpmag': 'bp_error', 'e_rpmag': 'rp_error'})
    gaia_cat = gaia_cat[['ra', 'dec', 'ra_error', 'dec_error', 'pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'g', 'bp', 'rp', 'g_error', 'bp_error', 'rp_error']]

    gaia_path = args.wd.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    np.save(gaia_path, gaia_cat.to_records(index=False))

    logger.info("Running pmfit with polynomial of degree {} as relative astrometric transformation.".format(args.degree))
    run_and_log(["pmfit", driver_path, "-d", str(args.degree), "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--mu-max=20"], logger=logger)

    logger.info("Running scene modeling")
    smphot_output = cwd.joinpath("smphot_output")
    smphot_output.mkdir(exist_ok=True)
    run_and_log(["mklc", "-t", cwd.joinpath("pmfit"), "-O", smphot_output, "-v", driver_path], logger=logger)

    return True


poloka_func.append({'reduce': smphot})


def smphot_plot(cwd, ztfname, filtercode, logger):
    logger.info("Running pmfit plots")
    driver_path = cwd.joinpath("{}_driver_{}".format(ztfname, filtercode))
    gaia_path = args.wd.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--plot", "--mu-max=20."], logger=logger)

    logger.info("Running smphot plots")
    with open(cwd.joinpath("smphot_output/lightcurve_sn.dat"), 'r') as f:
        _, sn_flux_df = list_format.read_list(f)

    plt.errorbar(sn_flux_df['mjd'], sn_flux_df['flux'], yerr=sn_flux_df['varflux'], fmt='.k')
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title("Calibrated lightcurve - {} - {}".format(ztfname, filtercode))
    plt.grid()
    plt.savefig(cwd.joinpath("{}-{}_smphot_lightcurve.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

    return True

poloka_func.append({'reduce': smphot_plot})

poloka_func = dict(zip([list(func.values())[0].__name__ for func in poloka_func], poloka_func))

scratch_files_to_ignore = ["output.log"]
def map_op(quadrant, wd, ztfname, filtercode, func, scratch=None):
    start_time = time.perf_counter()
    quadrant_dir = wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))

    logger = None
    if func != 'clean':
        logger = logging.getLogger(quadrant)
        logger.addHandler(logging.FileHandler(str(quadrant_dir.joinpath("output.log")), mode='a'))
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

            [shutil.copyfile(f, quadrant_scratch.joinpath(f.name)) for f in files if f.name not in scratch_files_to_ignore]
            quadrant_dir = quadrant_scratch
            logger.info("Successfully copyed files from sps to scratchspace")

    result = False
    try:
        result = poloka_func[func]['map'](quadrant_dir, logger)
    except Exception as e:
        logger.error("")
        logger.error("In folder {}".format(quadrant_dir))
        logger.error(traceback.format_exc())
        print(traceback.format_exc())
    finally:
        if scratch and func != 'clean':
            logger.info("Erasing quadrant data from scratchspace")
            files = list(quadrant_dir.glob("*"))
            [shutil.copy2(f, wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))) for f in files]
            [f.unlink() for f in files]
            quadrant_dir.rmdir()

    if func != 'clean':
        logger.info("Done.")

    return result, time.perf_counter(), start_time, get_worker().id


def reduce_op(results, cwd, ztfname, filtercode, func, save_stats):
    folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))

    # If we want to agregate run statistics on the previous map operation
    if save_stats and results is not None and any(results) and func != 'clean':
        results_df = pd.DataFrame([result for result in results if result is not None], columns=['result', 'time_end', 'time_start', 'worker_id'])
        results_df.to_csv(folder.joinpath("results_{}.csv".format(func)), index=False)

    if not 'reduce' in poloka_func[func].keys():
        return

    logger = logging.getLogger("{}-{}".format(ztfname, filtercode))
    logger.addHandler(logging.FileHandler(folder.joinpath("output.log"), mode='a'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())
    logger.info("Running reduction {}".format(args.func))

    start_timer = time.perf_counter()
    try:
        result = poloka_func[func]['reduce'](folder, ztfname, filtercode, logger)
    except Exception as e:
        logger.error("")
        logger.error("In SN {}-{}".format(ztfname, filtercode))
        logger.error(traceback.format_exc())
    finally:
        pass


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
    argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path)
    argparser.add_argument('--log-results', action='store_true', default=True)
    argparser.add_argument('--degree', type=int, default=3, help="Degree of polynomial for relative astrometric fit in pmfit.")

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

    print("Found {} SN1a".format(len(ztfnames)))

    if args.scratch:
        args.scratch.mkdir(exist_ok=True, parents=True)

        import signal
        import atexit
        def delete_scratch_at_exit(scratch_dir):
            shutil.rmtree(scratch_dir)

        atexit.register(delete_scratch_at_exit, scratch_dir=args.scratch)


    if args.cluster:
        # cluster = SLURMCluster(cores=12,
        #                        processes=12,
        #                        memory="32GB",
        #                        project="ztf",
        #                        walltime="12:00:00",
        #                        queue="htc",
        #                        job_extra=["-L sps"])

        cluster = SGECluster(cores=5,
                             processes=5,
                             queue="long",
                             memory="20GB",
                             #project="ztf",
                             walltime="12:00:00",
                             job_extra=["-l sps=1"])

        cluster.scale(jobs=120)
        client = Client(cluster)
        print(client.dashboard_link, flush=True)
        print(socket.gethostname(), flush=True)
        client.wait_for_workers(1)
    else:
        if args.n_jobs == 1:
            dask.config.set(scheduler='synchronous')

        localCluster = LocalCluster(n_workers=args.n_jobs, dashboard_address='localhost:8787', threads_per_worker=1, nanny=False)
        client = Client(localCluster)
        print("Dask dashboard at: {}".format(client.dashboard_link))


    jobs = []
    quadrant_count = 0
    reduction_count = 0
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            print("Building job list for {}-{}... ".format(ztfname, filtercode), end="", flush=True)

            quadrants_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not quadrants_folder.exists():
                print("No quadrant found.")
                continue

            results = None

            if 'map' in poloka_func[args.func].keys() and not args.no_map:
                quadrants = list(map(lambda x: x.stem, filter(lambda x: x.is_dir(), quadrants_folder.glob("ztf*"))))
                quadrant_count += len(quadrants)

                results = [delayed(map_op)(quadrant, args.wd, ztfname, filtercode, args.func, args.scratch) for quadrant in quadrants]
                print("Found {} quadrants. ".format(len(quadrants)), end="", flush=True)

            if ('reduce' in poloka_func[args.func].keys() or args.log_results) and not args.no_reduce:
                results = [delayed(reduce_op)(results, args.wd, ztfname, filtercode, args.func, True)]
                reduction_count += 1
                print("Found reduction.", end="", flush=True)

            print("")

            if results:
                jobs.extend(results)

    print("")
    print("Running. ", end="")

    if quadrant_count > 0:
        print(" Processing {} quadrants.".format(quadrant_count))

    if reduction_count > 0:
        print(" Processing {} reductions.".format(reduction_count))

    start_time = time.perf_counter()
    fjobs = client.compute(jobs)
    wait(fjobs)
    print("Done. Elapsed time={}".format(time.perf_counter() - start_time))

    client.close()
