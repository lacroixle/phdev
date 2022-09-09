#!/usr/bin/env python3

import subprocess
import time

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
