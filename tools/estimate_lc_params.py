#!/usr/bin/env python3
import logging
import sys
import os
import pathlib
import distutils.util

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


ztfname = sys.argv[1]
t0_inf = int(sys.argv[2])
t0_sup = int(sys.argv[3])
off_mul = float(sys.argv[4])

if ztfname != pathlib.Path(ztfname).stem:
    ztfname = str(pathlib.Path(ztfname).stem).split("_")[0]

if len(sys.argv) > 5:
    plot = bool(distutils.util.strtobool(sys.argv[5]))
    if not plot:
        save_folder = pathlib.Path(sys.argv[6]).resolve()


zmax = 2


def extract_interval(ztfname, t0_inf, t0_sup, off_mul):
    data_folder = pathlib.Path(os.environ.get("DATA_FOLDER"))
    lc_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/lightcurves/{}_LC.csv".format(ztfname)), delimiter="\s+", index_col="mjd")
    salt_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/params/DR2_SALT2fit_params.csv"), delimiter=",", index_col="name")

    lc_df = lc_df[(np.abs(stats.zscore(lc_df['flux_err'])) < zmax)]

    t_0 = salt_df.loc[ztfname, "t0"]

    t_inf = t_0 - t0_inf
    t_sup = t_0 + t0_sup

    def _compute_min_max_interval(lc_df, t_inf, t_sup, filt):
        lc_f_df = lc_df.loc[lc_df['filter'] == 'ztf{}'.format(filt[-1])]

        obs_count = len(lc_f_df[t_inf:t_sup]) - 1

        if obs_count == -1:
            return None

        idx_min = max(0, int(len(lc_f_df[:t_inf]) - off_mul*obs_count) - 1)
        idx_max = min(len(lc_f_df) - 1, int(len(lc_f_df[:t_sup]) + off_mul*obs_count))

        t_min = lc_f_df.iloc[idx_min].name
        t_max = lc_f_df.iloc[idx_max].name

        return {'lc': lc_f_df.loc[t_min:t_max], 't_min': t_min, 't_max': t_max}


    return dict([(filt, _compute_min_max_interval(lc_df, t_inf, t_sup, filt)) for filt in ['zr', 'zg','zi']]), t_inf, t_sup, t_0

lc_dict, t_inf, t_sup, t_0 = extract_interval(ztfname, t0_inf, t0_sup, off_mul)
print(".", flush=True, end="")

def plot_obs_count(ax, lc_df, t_0, t_inf, t_sup):
    ax.text(0., 0.15, str(len(lc_df.loc[:t_inf])), fontsize=15, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
    ax.text(t_0, 0.15, str(len(lc_df.loc[t_inf:t_sup])), fontsize=15, transform=ax.get_xaxis_transform(), horizontalalignment='left', verticalalignment='top')
    ax.text(1., 0.15, str(len(lc_df.loc[t_sup:])), fontsize=15, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    ax.axvline(t_inf)
    ax.axvline(t_0, linestyle='--')
    ax.axvline(t_sup)


with plt.style.context('seaborn-whitegrid'):

    fig, ax = plt.subplots(figsize=(15, 8), nrows=3, ncols=1, sharex=True)
    if lc_dict['zg']:
        ax = plt.subplot(3, 1, 1)
        lc_dict['zg']['lc']['flux'].plot(ax=ax, yerr=lc_dict['zg']['lc']['flux_err'], linestyle='None', marker='.', color='blue')
        plot_obs_count(ax, lc_dict['zg']['lc'], t_0, t_inf, t_sup)
        ax.grid(ls='--', linewidth=0.8)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux - g")

        ax.set_title(ztfname)

    if lc_dict['zr']:
        ax = plt.subplot(3, 1, 2)
        lc_dict['zr']['lc']['flux'].plot(ax=ax, yerr=lc_dict['zr']['lc']['flux_err'], linestyle='None', marker='.', color='red')
        plot_obs_count(ax, lc_dict['zr']['lc'], t_0, t_inf, t_sup)
        ax.grid(ls='--', linewidth=0.8)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux - r")


    if lc_dict['zi']:
        ax = plt.subplot(3, 1, 3)
        lc_dict['zi']['lc']['flux'].plot(ax=ax, yerr=lc_dict['zi']['lc']['flux_err'], linestyle='None', marker='.', color='orange')
        plot_obs_count(ax, lc_dict['zi']['lc'], t_0, t_inf, t_sup)
        ax.grid(ls='--', linewidth=0.8)
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux - i")

    plt.tight_layout()

    if plot:
        plt.show()
    else:
        plt.savefig(save_folder.joinpath("{}/{}".format(save_folder, ztfname)).with_suffix(".png"), dpi=300)
        plt.savefig(save_folder.joinpath("{}/{}".format(save_folder, ztfname)).with_suffix(".pdf"), dpi=300)


def _fix_ipac_file(filename):
    split = filename.split(".")
    return "{}_{}.fits".format(split[0], split[1])


def generate_lc_df(lc_dict, zfilter):
    if lc_dict[zfilter] is not None:
        lc_dict[zfilter]['lc']['ipac_file'] = lc_dict[zfilter]['lc']['ipac_file'].apply(_fix_ipac_file)
        return lc_dict[zfilter]['lc']['ipac_file']


def generate_params_df(lc_dict, zfilter):
    if lc_dict[zfilter] is not None:
        params = {'t_0': t_0,
                't_inf': t_inf,
                't_sup': t_sup,
                't_min': lc_dict[zfilter]['t_min'],
                't_max': lc_dict[zfilter]['t_max'],
                'zmax': zmax,
                'off_mul': off_mul}

        return pd.DataFrame.from_records([params])


def save_df(df_lc_zg, df_lc_zr, df_lc_zi, df_params_zg, df_params_zr, df_params_zi):
    def _save_df_filter(df_lc, df_params, zfilter, first=False):
        if df_lc is not None and df_params is not None:
            if first:
                mode = 'w'
            else:
                mode = 'a'

            df_lc.to_csv(save_folder.joinpath("{}_{}.csv".format(ztfname, zfilter)), sep=",")
            df_params.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, zfilter)), sep=",")
            df_lc.to_hdf(save_folder.joinpath("{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter), mode=mode)
            df_params.to_hdf(save_folder.joinpath("{}.hd5".format(ztfname)), key='params_{}'.format(zfilter))

    #pd.DataFrame().to_hdf(save_folder.joinpath("{}.hd5".format(ztfname)), key='void', mode='w')
    _save_df_filter(df_lc_zg, df_params_zg, 'zg', first=True)
    _save_df_filter(df_lc_zr, df_params_zr, 'zr')
    _save_df_filter(df_lc_zi, df_params_zi, 'zi')

    # if df_lc_zr is not None:
    #     df_lc_zr.to_csv(save_folder.joinpath("{}_{}.csv".format(ztfname, 'zr')), sep=",")
    #     df_params_zi.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, 'zr')), sep=",")
    #     df_lc_zr.to_hdf("{}.hd5".format(ztfname), key='lc_zg', mode='w')
    #     df_params_zr.to_hdf("{}.hd5".format(ztfname), key='params_zg')

    # if df_lc_zi is not None:
    #     df_lc_zi.to_csv(save_folder.joinpath("{}_{}.csv".format(ztfname, 'zg')), sep=",")
    #     df_params_zi.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, 'zg')), sep=",")
    #     df_lc_zi.to_hdf(save_folder.joinpath("{}.hd5".format(ztfname)), key='lc_zg', mode='w')
    #     df_params_zi.to_hdf(save_folder.joinpath("{}.hd5".format(ztfname)), key='params_zg')


    # df_lc_zr.to_csv(save_folder.joinpath("{}_{}.csv".format(ztfname, 'zr')), sep=",")
    # df_lc_zi.to_csv(save_folder.joinpath("{}_{}.csv".format(ztfname, 'zi')), sep=",")
    # df_params_zg.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, 'zg')), sep=",")
    # df_params_zr.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, 'zr')), sep=",")
    # df_params_zi.to_csv(save_folder.joinpath("{}_{}_params.csv".format(ztfname, 'zi')), sep=",")

    # df_lc_zr.to_hdf("{}.hd5".format(ztfname), key='lc_zr')
    # df_lc_zi.to_hdf("{}.hd5".format(ztfname), key='lc_zi')
    # df_params_zr.to_hdf("{}.hd5".format(ztfname), key='params_zr')
    # df_params_zi.to_hdf("{}.hd5".format(ztfname), key='params_zi')

save_df(generate_lc_df(lc_dict, 'zg'),
        generate_lc_df(lc_dict, 'zr'),
        generate_lc_df(lc_dict, 'zi'),
        generate_params_df(lc_dict, 'zg'),
        generate_params_df(lc_dict, 'zr'),
        generate_params_df(lc_dict, 'zi'))
