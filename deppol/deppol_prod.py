#!/usr/bin/env python3

import argparse
import pathlib
import sys
import logging

from deppol_utils import run_and_log
import utils


def generate_jobs(wd, run_folder, func, run_name):
    print("Working directory: {}".format(wd))
    print("Run folder: {}".format(run_folder))
    print("Func list: {}".format(func))

    print("Saving jobs under {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))

    batch_folder.mkdir(exist_ok=True)
    log_folder.mkdir(exist_ok=True)

    print("Generating jobs...")
    sne_jobs = {}
    job_count = 0
    for sn_folder in list(wd.glob("*")):
        filtercodes = []
        for filtercode in utils.filtercodes:
            if sn_folder.joinpath(filtercode).exists():
                filtercodes.append(filtercode)

        if len(filtercodes) > 0:
            sne_jobs[sn_folder.name] = filtercodes
            job_count += len(filtercodes)

    print("Job count: {}".format(job_count))

    for ztfname in sne_jobs.keys():
        for filtercode in sne_jobs[ztfname]:
            job = """#!/bin/sh
source ~/pyenv/bin/activate
export PYTHONPATH=${{PYTHONPATH}}:~/phdev/tools
export PATH=${{PATH}}:~/phdev/deppol
ulimit -n 4096
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={} --filtercode={} -j 4 --wd={} --func={} --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=/tmp/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10.
""".format(ztfname, filtercode, wd, ",".join(func))
            with open(batch_folder.joinpath("{}-{}.sh".format(ztfname, filtercode)), 'w') as f:
                f.write(job)


def schedule_jobs(run_folder, run_name):
    print("Run folder: {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))

    logger = logging.getLogger("schedule_jobs")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    batches = list(batch_folder.glob("*.sh"))
    for batch in batches[:10]:
        batch_name = batch.name.split(".")[0]
        cmd = ["sbatch", "--ntasks=4",
               "-D", "{}".format(run_folder.joinpath(run_name)),
               "-J", "{}_smp".format(batch_name),
               "-o", log_folder.joinpath("log_{}".format(batch_name)),
               "-A", "ztf",
               "-L", "sps",
               #"--spread-job",
               batch]
        #print(" ".join(map(lambda x: str(x), cmd)))
        returncode = run_and_log(cmd, logger)
        print("{}: {}".format(batch_name, returncode))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--generate-jobs', action='store_true', help="If set, generate list of jobs")
    argparser.add_argument('--schedule-jobs', action='store_true', help="If set, schedule jobs onto SLURM")
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--func', type=pathlib.Path, help="")
    argparser.add_argument('--run-name', type=str, required=True)

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()

    if not args.wd.exists():
        sys.exit("Working folder does not exist!")

    if not args.run_folder.exists():
        sys.exit("Run folder does not exist!")

    funcs = []
    if args.func.exists():
        with open(args.func, 'r') as f:
            funcs = list(map(lambda x: x.strip(), f.readlines()))
    else:
        funcs = str(args.func).split(",")

    args.run_folder.joinpath(args.run_name).mkdir(exist_ok=True)

    if args.generate_jobs:
        generate_jobs(args.wd, args.run_folder, funcs, args.run_name)

    if args.schedule_jobs:
        schedule_jobs(args.run_folder, args.run_name)
