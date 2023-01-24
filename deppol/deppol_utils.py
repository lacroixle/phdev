#!/usr/bin/env python3

import pathlib
import subprocess
import time
from collections.abc import Iterable
import json


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


def dump_timings(start_time, end_time, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps({'start': start_time, 'end': end_time, 'elapsed': end_time-start_time}))


def ztfnames_from_string(ztfname):
    if ztfname is not None:
        if ztfname.stem == str(ztfname):
            return [str(ztfname)]
        else:
            ztfname = ztfname.expanduser().resolve()
            if ztfname.exists():
                with open(ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
                return ztfnames
            else:
                pass


def lc_folder_args(args):
    return args.lc_folder.expanduser().resolve()


def quadrants_from_band_path(band_path, logger, check_files=None, paths=True, ignore_noprocess=False):
    if not ignore_noprocess:
        noprocess = noprocess_quadrants(band_path)
    else:
        noprocess = []

    quadrant_paths = [quadrant_path for quadrant_path in list(band_path.glob("ztf_*")) if quadrant_path.name not in noprocess]

    if check_files:
        if not isinstance(check_files, Iterable) or isinstance(check_files, str) or isinstance(check_files, pathlib.Path):
            check_files = [check_files]

        def _check_files(quadrant_path):
            check_ok = True
            for check_file in check_files:
                if not quadrant_path.joinpath(check_file).exists():
                    check_ok = False
                    break

            return check_ok

        quadrant_paths = list(filter(_check_files, quadrant_paths))

    if paths:
        return quadrant_paths
    else:
        return [quadrant_path.name for quadrant_path in quadrant_paths]


def noprocess_quadrants(band_path):
    noprocess = []
    if band_path.joinpath("noprocess").exists():
        with open(band_path.joinpath("noprocess"), 'r') as f:
            for line in f.readlines():
                quadrant = line.strip()
                if quadrant[0] == "#":
                    continue
                elif band_path.joinpath(quadrant).exists():
                    noprocess.append(quadrant)

    return noprocess
