#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime

from joblib import Parallel, delayed


filtercodes = ['zg', 'zr', 'zi']

argparser = argparse.ArgumentParser(description="")
argparser.add_argument('--ztfname', type=str, required=True)
argparser.add_argument('-j', dest=n_jobs, type=int, default=1)
argparser.add_argument('--wd', type=pathlib.Path, help="Working directory")
argparser.add_argument('--filtercode', choices=filtercodes)
argparser.add_argument('--func')

args.argparser.parse_args()

def run_and_log(cmd, logger=None):
    out = subprocess.run(cmd, capture_output=True)
    if logger:
        logger.info(out.stdout)
        logger.error(out.stderr)

    return out.returncode


def make_catalog(folder, logger):

    run_and_log(folder, logger)
    return folder.joinpath("se.list").exists()


def mkcat2(folder, logger):
    pass


def makepsf(folder, logger):
    pass


def all_cmd(folder):

    if not make_catalog(folder):
        pass

    if not mkcat2(folder):
        pass

    if not makepsf(folder):
        pass


def launch(folder, fct):
    logger = logging.getLogger(folder.name)
    logger.addHandler(logging.FileHandler(folder.joinpath("output.log"), mode='w'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())

    fct(folder, logger)

    print(".", end="", flush=True)
    logger.info("Done.")

folders = args.wd.glob()

results = Parallel(n_jobs=n_jobs)(delayed(launch)(folder, make_catalog) for folder in foldersfct)

[print(folder) for folder, result in zip(folders, results) if result]
