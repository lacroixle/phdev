#!/usr/bin/env python3
import sys

import pandas as pd
import pathlib
from ztfimg import science
from joblib import Parallel, delayed

ztfname = sys.argv[1]
lc_folder = pathlib.Path(sys.argv[2])

n_jobs = 1
if len(sys.argv) > 3:
    n_jobs = int(sys.argv[3])


print("Downloading quadrants for {}".format(ztfname))
print("Running {} threads...".format(n_jobs))

def download_lc(ztfname, filter_key):
    lc_df = pd.read_hdf(lc_folder.joinpath("{}.hd5".format(ztfname)), key=filter_key)

    print("Downloading filter r ({} quadrants)".format(lc_df.size))
    print("Estimated size to download=~{} GB".format(lc_df.size*37*2/1024))

    def _download_lc(lc_filename):
        print(".", end="", flush=True)
        science.ScienceQuadrant.from_filename(lc_filename)

    #for lc_filename in lc_df.to_list():
    #    _download_lc(lc_filename)

    Parallel(n_jobs=n_jobs)(delayed(_download_lc)(lc_filename) for lc_filename in lc_df.to_list())

    print("\n", end="")

download_lc(ztfname, 'lc_zr')
download_lc(ztfname, 'lc_zg')
download_lc(ztfname, 'lc_zi')
