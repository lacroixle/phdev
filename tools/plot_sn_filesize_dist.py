#!/usr/bin/env python3

import sys
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

lc_folder = pathlib.Path(sys.argv[1])

sciimg_size = 38
mskimg_size = 18

# Retrieve all SN interval parameter files
param_files = list(lc_folder.glob("*.hd5"))


def sn_filesize_zf(zfilter):
    filesizes = []

    for param_file in param_files:
        with pd.HDFStore(param_file, mode='r') as store:
            if '/lc_{}'.format(zfilter) in store.keys():
                lc_df = pd.read_hdf(store, key='lc_{}'.format(zfilter))
                filesizes.append(len(lc_df)*(sciimg_size+mskimg_size)*1e-3)

    return filesizes



sizes = Parallel(n_jobs=3)(delayed(sn_filesize_zf)(zfilter) for zfilter in ['zr', 'zg', 'zi'])


# sizes_zr = sn_filesize_zf('zr')
# sizes_zg = sn_filesize_zf('zg')
# sizes_zi = sn_filesize_zf('zi')

bins = 200

plt.subplot(4, 1, 1)
plt.hist(sizes[0], bins, range=[0, 60])
plt.ylabel("ztfr")

plt.subplot(4, 1, 2)
plt.hist(sizes[1], bins, range=[0, 60])
plt.ylabel("ztfg")

plt.subplot(4, 1, 3)
plt.hist(sizes[2], bins, range=[0, 60])
plt.ylabel("ztfi")

plt.subplot(4, 1, 4)
plt.hist([sum(size) for size in zip(*sizes)], bins, range=[0, 60])
plt.xlabel("GB/sn")
plt.ylabel("sum")

plt.show()
