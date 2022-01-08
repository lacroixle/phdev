#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime
import os

from joblib import Parallel, delayed


filtercodes = ['zg', 'zr', 'zi']
poloka_fct = []


def run_and_log(cmd, logger=None):
    out = subprocess.run(cmd, capture_output=True)
    if logger:
        logger.info(out.stdout.decode('utf-8'))
        logger.error(out.stderr.decode('utf-8'))

    return out.returncode

poloka_fct.append(run_and_log)


def make_catalog(folder, logger):
    run_and_log(["make_catalog", folder], logger)
    return folder.joinpath("se.list").exists()

poloka_fct.append(make_catalog)


def mkcat2(folder, logger):
    run_and_log(["mkcat2", folder], logger)
    return folder.joinpath("standalone_stars.list").exists()

poloka_fct.append(mkcat2)


def makepsf(folder, logger):
    run_and_log(["makepsf", folder], logger)
    return folder.joinpath("psfstars.list").exists()

poloka_fct.append(makepsf)


def pipeline(folder, logger):
    if not make_catalog(folder, logger):
        return False

    if not mkcat2(folder, logger):
        return False

    if not makepsf(folder, logger):
        return False

    return True

poloka_fct.append(pipeline)

poloka_fct = dict(zip([fct.__name__ for fct in poloka_fct], poloka_fct))


def launch(folder, fct):
    logger = logging.getLogger(folder.name)
    logger.addHandler(logging.FileHandler(folder.joinpath("output.log"), mode='w'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())

    result = fct(folder, logger)

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
argparser.add_argument('--func', choices=poloka_fct.keys(), default='pipeline')

args = argparser.parse_args()


print("Running Poloka function {} on {}-{}".format(args.func, args.ztfname, args.filtercode))

# First check if there is quadrants associated with the selected filter
if not args.wd.joinpath("{}/{}".format(args.ztfname, args.filtercode)).exists():
    print("No quadrant for filter {}! Skipping.".format(args.filtercode))

cwd = args.wd.joinpath("{}/{}".format(args.ztfname, args.filtercode))
os.chdir(cwd)

folders = list(cwd.glob("*"))

print("Found {} quadrant folder.".format(len(folders)))
results = Parallel(n_jobs=args.n_jobs)(delayed(launch)(folder, wholeseq) for folder in folders)

print("")

# Print quadrant that failed
if any not results:
    [print(folder) for folder, result in zip(folders, results) if not result]

    print("")
