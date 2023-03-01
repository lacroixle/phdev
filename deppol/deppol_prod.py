#!/usr/bin/env python3

import argparse
import pathlib
import sys
import logging
import subprocess
import shutil

import pandas as pd

from deppol_utils import run_and_log, load_timings
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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 deppol --ztfname={} --filtercode={} -j {j} --wd={} --func={} --lc-folder=/sps/ztf/data/storage/scenemodeling/lc --quadrant-workspace=/dev/shm/llacroix --rm-intermediates --scratch=${{TMPDIR}}/llacroix --astro-degree=5 --max-seeing=4. --discard-calibrated --astro-min-mag=-10. --dump-node-info --from-scratch --dump-timings --parallel-reduce
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

    if args.ztfname:
        if args.ztfname.exists():
            with open(args.ztfname, 'r') as f:
                lines = f.readlines()

            batches = [batch_folder.joinpath("{}.sh".format(line.strip())) for line in lines]
            for batch in batches:
                if not batch.exists():
                    print("{} does not exist!".format(batch))
                    exit()

            print("Scheduling {} jobs".format(len(batches)))
        else:
            ztfbatch = batch_folder.joinpath("{}.sh".format(args.ztfname))
            if ztfbatch.exists():
                batches = [ztfbatch]
            else:
                print("--ztfname specified but does not correspond to a list or a sn-filtercode!")
                exit()


    for batch in batches:
        batch_name = batch.name.split(".")[0]
        if batch_name in scheduled_jobs.keys():
            continue

        batch_status_path = status_folder.joinpath(batch_name)
        if batch_status_path.exists() and not args.ztfname:
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
               "--mem={}G".format(3*args.j),
               "-t", "3-0",
               batch]

        returncode = run_and_log(cmd, logger)

        with open(batch_status_path, 'w') as f:
            if returncode == 0:
                f.write("scheduled")
            else:
                break
                f.write("failedtoschedule")
        print("{}: {}".format(batch_name, returncode))


def generate_summary(args, funcs):
    def __is_job_done(status_name):
        with open(status_name, 'r') as f:
            return "done" in f.readline()

    def __func_status(folder, func):
        if folder.joinpath("{}.success".format(func)).exists():
            return True
        elif folder.joinpath("{}.fail".format(func)).exists():
            return False
        else:
            return None

    status_folder = args.run_folder.joinpath("{}/status".format(args.run_name))
    sne_status = list(status_folder.glob("*"))

    ztfname_filter_list = [sn_status.name for sn_status in sne_status if __is_job_done(sn_status)]
    func_status = {}
    func_timings = {}
    failed_list = []
    success_sne = 0
    success_smp_sn = 0
    success_smp_stars = 0
    for ztfname_filter in ztfname_filter_list:
        ztfname, filtercode = ztfname_filter.split("-")
        band_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
        func_status[ztfname_filter] = {}
        func_timings[ztfname_filter] = {}
        func_timings[ztfname_filter]['quadrant_count'] = len(list(band_folder.glob("ztf_*")))

        # Get number of workers from the log...
        with open(args.run_folder.joinpath("{}/logs/log_{}".format(args.run_name, ztfname_filter))) as f:
            worker_count = -1
            for i in range(200):
                line = f.readline().strip()
                if "Running a local cluster with" in line:
                    worker_count = int(line.split(" ")[5])

        func_timings[ztfname_filter]['worker_count'] = worker_count
        success = True
        for func in funcs:
            status = __func_status(band_folder, func)
            func_status[ztfname_filter][func] = status

            if band_folder.joinpath("timings_{}".format(func)).exists():
                elapsed = load_timings(band_folder.joinpath("timings_{}".format(func)))['total']['elapsed']
                func_timings[ztfname_filter][func] = elapsed

        if not func_status[ztfname_filter]['smphot'] or not func_status[ztfname_filter]['smphot_stars']:
            failed_list.append(ztfname_filter)
        else:
            success_sne += 1

        if func_status[ztfname_filter]['smphot']:
            success_smp_sn += 1

        if func_status[ztfname_filter]['smphot_stars']:
            success_smp_stars += 1

        if not args.print_failed:
            print(".", end="", flush=True)

    if not args.print_failed:
        print("Run summary:")
        print("Total SNe={}".format(len(ztfname_filter_list)))
        print("Success={}".format(success_sne))
        print("Failed={}".format(len(failed_list)))
        print("Success SN SMP={}".format(success_smp_sn))
        print("Success SN stars={}".format(success_smp_stars))

    df_status = pd.DataFrame.from_dict(func_status, orient='index')
    df_status.to_csv("pipeline_status.csv")

    df_timings = pd.DataFrame.from_dict(func_timings, orient='index')
    df_timings.to_csv("pipeline_timings.csv")

    if args.print_failed:
        for failed in failed_list:
            print(failed)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--generate-jobs', action='store_true', help="If set, generate list of jobs")
    argparser.add_argument('--schedule-jobs', action='store_true', help="If set, schedule jobs onto SLURM")
    argparser.add_argument('--wd', type=pathlib.Path, required=False)
    argparser.add_argument('--run-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--func', type=pathlib.Path, help="")
    argparser.add_argument('--run-name', type=str, required=True)
    argparser.add_argument('-j', default=1, type=int)
    argparser.add_argument('--purge-status', action='store_true')
    argparser.add_argument('--purge', action='store_true')
    argparser.add_argument('--ztfname', type=pathlib.Path, help="To schedule only one SN. Followong ZTFSNname-filtercode, eg ZTF19aaripqw-zg.\nTo schedule a list, put a filename (as for deppol).\nForces through status.")
    argparser.add_argument('--generate-summary', action='store_true')
    argparser.add_argument('--print-failed', action='store_true')

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

    if args.generate_summary:
        generate_summary(args, funcs)

    if args.generate_jobs:
        generate_jobs(args.wd, args.run_folder, funcs, args.run_name)

    if args.schedule_jobs:
        schedule_jobs(args.run_folder, args.run_name)
