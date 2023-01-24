#!/usr/bin/env python3

import argparse
import pathlib
import sys
import logging
import subprocess
import shutil

from deppol_utils import run_and_log
import utils


def generate_jobs(wd, run_folder, func, run_name):
    print("Working directory: {}".format(wd))
    print("Run folder: {}".format(run_folder))
    print("Func list: {}".format(func))

    print("Saving jobs under {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))
    status_folder = run_folder.joinpath("{}/status".format(run_name))

    batch_folder.mkdir(exist_ok=True)
    log_folder.mkdir(exist_ok=True)
    status_folder.mkdir(exist_ok=True)

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
echo "running" > {status_path}
source ~/pyenv/bin/activate
export PYTHONPATH=${{PYTHONPATH}}:~/phdev/tools
export PATH=${{PATH}}:~/phdev/deppol
ulimit -n 4096
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={} --filtercode={} -j {j} --wd={} --func={} --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=${{TMPDIR}}/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10. --dump-node-info --from-scratch --dump-timings --log-overwrite
echo "done" > {status_path}
""".format(ztfname, filtercode, wd, ",".join(func), status_path=run_folder.joinpath("{}/status/{}-{}".format(run_name, ztfname, filtercode)), j=args.j)
            with open(batch_folder.joinpath("{}-{}.sh".format(ztfname, filtercode)), 'w') as f:
                f.write(job)


def schedule_jobs(run_folder, run_name):
    print("Run folder: {}".format(run_folder))
    batch_folder = run_folder.joinpath("{}/batches".format(run_name))
    log_folder = run_folder.joinpath("{}/logs".format(run_name))
    status_folder = run_folder.joinpath("{}/status".format(run_name))
    status_folder.mkdir(exist_ok=True)

    logger = logging.getLogger("schedule_jobs")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # First get list of currently scheduled jobs
    out = subprocess.run(["squeue", "-o", "%j,%t", "-p", "htc", "-h"], capture_output=True)
    scheduled_jobs_raw = out.stdout.decode('utf-8').split("\n")
    scheduled_jobs = dict([(scheduled_job.split(",")[0][4:], scheduled_job.split(",")[1]) for scheduled_job in scheduled_jobs_raw if scheduled_job[:4] == "smp_"])

    batches = list(batch_folder.glob("*.sh"))
    for batch in batches:
        batch_name = batch.name.split(".")[0]
        if batch_name in scheduled_jobs.keys():
            continue

        batch_status_path = status_folder.joinpath(batch_name)
        if batch_status_path.exists():
            with open(batch_status_path, 'r') as f:
                status = f.readline().strip()
                if status == "scheduled" or status == "done" or status == "running":
                    continue

        cmd = ["sbatch", "--ntasks={}".format(args.j),
               "-D", "{}".format(run_folder.joinpath(run_name)),
               "-J", "smp_{}".format(batch_name),
               "-o", log_folder.joinpath("log_{}".format(batch_name)),
               "-A", "ztf",
               "-L", "sps",
               batch]

        returncode = run_and_log(cmd, logger)

        with open(batch_status_path, 'w') as f:
            if returncode == 0:
                f.write("scheduled")
            else:
                break
                f.write("failedtoschedule")
        print("{}: {}".format(batch_name, returncode))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--generate-jobs', action='store_true', help="If set, generate list of jobs")
    argparser.add_argument('--schedule-jobs', action='store_true', help="If set, schedule jobs onto SLURM")
    argparser.add_argument('--wd', type=pathlib.Path, required=False)
    argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--func', type=pathlib.Path, help="")
    argparser.add_argument('--run-name', type=str, required=True)
    argparser.add_argument('-j', default=1)
    argparser.add_argument('--purge-status', action='store_true')

    args = argparser.parse_args()

    if args.wd:
        args.wd = args.wd.expanduser().resolve()
        if not args.wd.exists():
            sys.exit("Working folder does not exist!")

    if args.purge_status:
        status_folder = args.run_folder.joinpath("{}/status".format(args.run_name))
        if status_folder.exists():
            shutil.rmtree(status_folder)


    if not args.run_folder.exists():
        sys.exit("Run folder does not exist!")

    if args.func:
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
