#!/usr/bin/env python3
import sys
import time
import argparse
import pathlib
import warnings

import pandas as pd
from ztfimg import science
from joblib import Parallel, delayed


filtercodes = ['zr', 'zg', 'zi']


argparser = argparse.ArgumentParser(description="")
argparser.add_argument('--ztfname', type=str, help="", required=True)
argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path, help="", required=True)
argparser.add_argument('--filtercode', type=str, choices=filtercodes, help="")
argparser.add_argument('-j', dest='n_jobs', type=int, default=1, help="")
args = argparser.parse_args()

ztfname = args.ztfname
lc_folder = args.lc_folder
n_jobs = args.n_jobs
if args.filtercode:
    filtercodes = [args.filtercode]

# Image size in MB
sciimg_size = 38
mskimg_size = 18

print("Downloading quadrants for {}".format(ztfname))
print("Running {} threads...".format(n_jobs))

total_estimated_filesize = 0

def download_lc(hdfstore, filtercode):
    lc_df = pd.read_hdf(hdfstore, key=filtercode)

    estimated_filesize = len(lc_df)*(sciimg_size+mskimg_size)/1000
    global total_estimated_filesize
    total_estimated_filesize = total_estimated_filesize + estimated_filesize
    print("Downloading filter {} ({} quadrants)".format(filtercode, len(lc_df)))
    print("Estimated size to download=~{} GB".format(estimated_filesize))

    def _download_lc(lc_filename):
        try:
            science.ScienceQuadrant.from_filename(lc_filename).get_data('clean')
        except FileNotFoundError:
            print("x", end="", flush=True)
            return False
        except Exception as e:
            print(e)
            return False
        else:
            print(".", end="", flush=True)
            return True

    start_dl_filter_time = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(_download_lc)(lc_filename) for lc_filename in lc_df['ipac_file'].to_list())
    elapsed_time = time.perf_counter() - start_dl_filter_time

    print("")
    print("Elapsed time={} s".format(elapsed_time))
    print("Average download speed={} MB/s".format(estimated_filesize/elapsed_time*1000))
    print("")

    if not all(results):
        print("Quadrants that could not be downloaded:")
        [print(quadrant_filename) for quadrant_filename, result in zip(lc_df['ipac_file'], results) if not result]
        print("")


# Remove user warning which messes up my beautiful dot plotting
warnings.simplefilter('ignore')

start_dl_time = time.perf_counter()

for filtercode in filtercodes:
    with pd.HDFStore(lc_folder.joinpath("{}.hd5".format(ztfname)), mode='r') as hdfstore:
        if '/lc_{}'.format(filtercode) in hdfstore.keys():
            download_lc(hdfstore, 'lc_{}'.format(filtercode))
        else:
            print("No {} filter found for {}!".format(filtercode, ztfname))

total_elapsed_time = time.perf_counter() - start_dl_time
print("Total time elapsed={}".format(total_elapsed_time))
print("Average download speed={} MB/s".format(total_estimated_filesize/total_elapsed_time*1000))
