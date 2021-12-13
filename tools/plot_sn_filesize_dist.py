#!/usr/bin/env python3

import sys
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

lc_folder = pathlib.Path(sys.argv[1])

sciimg_size = 38
mskimg_size = 18
memory_multiplier = (sciimg_size + mskimg_size)*1e-3

# Retrieve all SN interval parameter files
param_files = list(lc_folder.glob("*.hd5"))


with_off_stat = True


def sn_filesize_zf(zfilter):
    filesizes = []

    for param_file in param_files:
        with pd.HDFStore(param_file, mode='r') as store:
            if '/lc_{}'.format(zfilter) in store.keys():
                lc_df = pd.read_hdf(store, key='lc_{}'.format(zfilter))
                if with_off_stat:
                    filesizes.append(lc_df.size*memory_multiplier)
                else:
                    lc_params_df = pd.read_hdf(store, key='params_{}'.format(zfilter))
                    filesizes.append(lc_df.loc[lc_params_df['t_inf'].item():lc_params_df['t_sup'].item()].size*memory_multiplier)

    return filesizes



sizes = Parallel(n_jobs=3)(delayed(sn_filesize_zf)(zfilter) for zfilter in ['zr', 'zg', 'zi'])


def compute_median(l):
    return np.sort(l)[int(len(l)/2)]


sum_sizes = [sum(size) for size in zip(*sizes)]

median_r = compute_median(sizes[0])
median_g = compute_median(sizes[1])
median_i = compute_median(sizes[2])
median_sum = compute_median(sum_sizes)
# media
# # nb de point dans le on

if with_off_stat:
    bins = 200
    max_x = 60
else:
    bins = 100
    max_x = 40

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)
fig.suptitle("SN size (in GB) distribution, OFF stat={}".format(with_off_stat))

ax = plt.subplot(4, 2, 1)
plt.hist(sizes[0], bins, range=[0, max_x], color='xkcd:dark grey', histtype='step')
plt.grid()
plt.axvline(median_r, linestyle='--', color='black')
plt.ylabel("ztfr")

plt.subplot(4, 2, 2)
plt.hist(sizes[0], bins, histtype='step', cumulative=True)

plt.subplot(4, 2, 3)
plt.hist(sizes[1], bins, range=[0, max_x], color='xkcd:dark grey', histtype='step')
plt.grid()
plt.axvline(median_g, linestyle='--', color='black')
plt.ylabel("ztfg")

plt.subplot(4, 2, 4)
plt.hist(sizes[1], bins, histtype='step', cumulative=True)

plt.subplot(4, 2, 5)
plt.hist(sizes[2], bins, range=[0, max_x], color='xkcd:dark grey', histtype='step')
plt.grid()
plt.axvline(median_i, linestyle='--', color='black')
plt.ylabel("ztfi")

plt.subplot(4, 2, 6)
plt.hist(sizes[2], bins, histtype='step', cumulative=True)

plt.subplot(4, 2, 7)
plt.hist(sum_sizes, bins, range=[0, max_x], color='xkcd:dark grey', histtype='step')
plt.grid()
plt.axvline(median_sum, linestyle='--', color='black')
plt.xlabel("GB/sn")
plt.ylabel("sum")

plt.subplot(4, 2, 8)
plt.hist(sum_sizes, bins, histtype='step', cumulative=True)

plt.show()
