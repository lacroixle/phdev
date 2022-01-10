#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime
import os
import time

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import list_format


filtercodes = ['zg', 'zr', 'zi']
poloka_fct = []


def run_and_log(cmd, logger=None):
    out = subprocess.run(cmd, capture_output=True)
    if logger:
        logger.info(out.stdout.decode('utf-8'))
        logger.error(out.stderr.decode('utf-8'))

    return out.returncode

poloka_fct.append({'map': run_and_log})


def make_catalog(folder, logger):
    run_and_log(["make_catalog", folder], logger)
    return folder.joinpath("se.list").exists()

poloka_fct.append({'map': make_catalog})


def mkcat2(folder, logger):
    run_and_log(["mkcat2", folder], logger)
    return folder.joinpath("standalone_stars.list").exists()

poloka_fct.append({'map': mkcat2})


def makepsf(folder, logger):
    run_and_log(["makepsf", folder], logger)
    return folder.joinpath("psfstars.list").exists()

poloka_fct.append({'map': makepsf})


def pipeline(folder, logger):
    if not make_catalog(folder, logger):
        return False

    if not mkcat2(folder, logger):
        return False

    if not makepsf(folder, logger):
        return False

    return True

poloka_fct.append({'map': pipeline})


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

poloka_fct.append({'map': clean})


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


def stats_reduce(cwd, ztfname, filtercode):
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

    plt.hist(seseeings, bins=60, range=[0.5, 3], color='xkcd:dark grey', histtype='step')
    plt.grid()
    plt.savefig(cwd.joinpath("{}-{}_seseeing_dist.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

poloka_fct.append({'map': stats, 'reduce': stats_reduce})


poloka_fct = dict(zip([fct['map'].__name__ for fct in poloka_fct], poloka_fct))


def launch(folder, fct):
    logger = logging.getLogger(folder.name)
    logger.addHandler(logging.FileHandler(folder.joinpath("output.log"), mode='w'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())

    try:
        result = fct['map'](folder, logger)
    except Exception as e:
        print("")
        print("In folder {}".format(folder))
        print(e.with_traceback)

    if not args.dry_run:
        if result:
            print(".", end="", flush=True)
        else:
            print("x", end="", flush=True)

    logger.info("Done.")
    return result


argparser = argparse.ArgumentParser(description="")
argparser.add_argument('--ztfname', type=str, required=True)
argparser.add_argument('-j', dest='n_jobs', type=int, default=1)
argparser.add_argument('--wd', type=pathlib.Path, help="Working directory")
argparser.add_argument('--filtercode', choices=filtercodes)
argparser.add_argument('--func', type=str, choices=poloka_fct.keys(), default='pipeline')
argparser.add_argument('--dry-run', dest='dry_run', action='store_true')
argparser.add_argument('--no-map', dest='no_map', action='store_true')
argparser.add_argument('--no-reduce', dest='no_reduce', action='store_true')

args = argparser.parse_args()
args.wd = args.wd.expanduser().resolve()

print("Running Poloka function {} on {}-{}".format(args.func, args.ztfname, args.filtercode))

# First check if there is quadrants associated with the selected filter
if not args.wd.joinpath("{}/{}".format(args.ztfname, args.filtercode)).is_dir():
    print("No quadrant for filter {}! Skipping.".format(args.filtercode))
    exit()

cwd = args.wd.joinpath("{}/{}".format(args.ztfname, args.filtercode))
os.chdir(cwd)

folders = [folder for folder in list(cwd.glob("*")) if folder.is_dir()]

print("Found {} quadrant folders".format(len(folders)))
if not args.no_map:

    map_start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs)(delayed(launch)(folder, poloka_fct[args.func]) for folder in folders)
    print("")
    print("Time elapsed={}".format(time.perf_counter() - map_start))

    # Print quadrant that failed
    if not all(results):
        [print(folder) for folder, result in zip(folders, results) if not result]
        print("")

if not args.no_reduce and 'reduce' in poloka_fct[args.func].keys():
    print("Reducing")
    reduce_start = time.perf_counter()
    poloka_fct[args.func]['reduce'](cwd, args.ztfname, args.filtercode)
    print("Time elapsed={}".format(time.perf_counter() - reduce_start))

print("")
