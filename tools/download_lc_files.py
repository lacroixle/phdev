#!/usr/bin/env python3
import sys
import time

import pandas as pd
import pathlib
from ztfimg import science
from joblib import Parallel, delayed

ztfname = sys.argv[1]
lc_folder = pathlib.Path(sys.argv[2])

n_jobs = 1
if len(sys.argv) > 3:
    n_jobs = int(sys.argv[3])

# Image size in MB
sciimg_size = 38
mskimg_size = 18

print("Downloading quadrants for {}".format(ztfname))
print("Running {} threads...".format(n_jobs))

total_estimated_filesize = 0

def download_lc(ztfname, filter_key):
    lc_df = pd.read_hdf(lc_folder.joinpath("{}.hd5".format(ztfname)), key=filter_key)

    estimated_filesize = lc_df.size*(sciimg_size+mskimg_size)/1000
    global total_estimated_filesize
    total_estimated_filesize = total_estimated_filesize + estimated_filesize
    print("Downloading filter r ({} quadrants)".format(lc_df.size))
    print("Estimated size to download=~{} GB".format(estimated_filesize))

    def _download_lc(lc_filename):
        try:
            science.ScienceQuadrant.from_filename(lc_filename).get_data('clean')
        except:
            print("x")
        else:
            print(".", end="", flush=True)

    start_dl_filter_time = time.perf_counter()
    Parallel(n_jobs=n_jobs)(delayed(_download_lc)(lc_filename) for lc_filename in lc_df.to_list())
    elapsed_time = time.perf_counter() - start_dl_filter_time

    print("")
    print("Elapsed time={} s".format(elapsed_time))
    print("Average download speed={} MB/s".format(estimated_filesize/elapsed_time*1000))
    print("")

start_dl_time = time.perf_counter()
download_lc(ztfname, 'lc_zr')
download_lc(ztfname, 'lc_zg')
download_lc(ztfname, 'lc_zi')
total_elapsed_time = time.perf_counter() - start_dl_time
print("Total time elapsed={}".format(total_elapsed_time))
print("Average download speed={} MB/s".format(total_estimated_filesize/total_elapsed_time*1000))
