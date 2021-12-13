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
from joblib import Parallel, delayed
from ztfquery import query
import astropy.time


ztfname_folder = pathlib.Path(sys.argv[1])
t0_inf = int(sys.argv[2])
t0_sup = int(sys.argv[3])
off_mul = float(sys.argv[4])


data_folder = pathlib.Path(os.environ.get("DATA_FOLDER"))
salt_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/params/DR2_SALT2fit_params.csv"), delimiter=",", index_col="name")
redshift_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/params/DR2_redshifts.csv"), delimiter=",", index_col="ztfname")

plot = True
n_jobs = 1
if len(sys.argv) > 5:
    plot = bool(distutils.util.strtobool(sys.argv[6]))
    if not plot:
        n_jobs = int(sys.argv[5])
        save_folder = pathlib.Path(sys.argv[7]).resolve()


ztfnames = []


if not ztfname_folder.is_dir():
    ztfnames = [str(ztfname_folder)]
else:
    ztf_files = ztfname_folder.glob("*_LC.csv")

    ztfnames = [ztf_file.stem.split("_")[0] for ztf_file in ztf_files]

# For some reason this sn does not exist in the SALT db
blacklist = ["ZTF18aaajrso"]

zmax = 5


def estimate_lc_params(ztfname):
    if ztfname in blacklist:
        return

    def extract_interval(ztfname, t0_inf, t0_sup, off_mul, do_sql_request=True):
        lc_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/lightcurves/{}_LC.csv".format(ztfname)), delimiter="\s+", index_col="mjd")
        lc_df = lc_df[(np.abs(stats.zscore(lc_df['flux_err'])) < zmax)]

        sql_lc_df = None
        if do_sql_request:
            zquery = query.ZTFQuery()
            zquery.load_metadata(radec=(redshift_df.loc[ztfname]['host_ra'], redshift_df.loc[ztfname]['host_dec']))
            sql_lc_df = zquery.metatable

            # Add an obsmkd column and set it as index
            def _jd_to_mjd(jd):
                time = astropy.time.Time(jd, format='jd')
                return time.mjd

            sql_lc_df['obsmjd'] = sql_lc_df['obsjd'].apply(_jd_to_mjd)
            sql_lc_df.set_index('obsmjd', inplace=True)

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

            return {'lc': lc_f_df.loc[t_min:t_max], 't_min': t_min, 't_max': t_max, 'sql_lc': sql_lc_df.loc[sql_lc_df['filtercode'] == filt]}


        return dict([(filt, _compute_min_max_interval(lc_df, t_inf, t_sup, filt)) for filt in ['zr', 'zg','zi']]), t_inf, t_sup, t_0


    lc_dict, t_inf, t_sup, t_0 = extract_interval(ztfname, t0_inf, t0_sup, off_mul)

    # Check that at least for one filter we have data
    if not any([lc_dict[zfilter] is not None for zfilter in ['zg', 'zr', 'zi']]):
        return

    def plot_obs_count(ax, lc_df, t_0, t_inf, t_sup):
        ax.text(0., 0.20, str(len(lc_df.loc[:t_inf])), fontsize=15, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
        ax.text(t_0, 0.20, str(len(lc_df.loc[t_inf:t_sup])), fontsize=15, transform=ax.get_xaxis_transform(), horizontalalignment='left', verticalalignment='top')
        ax.text(1., 0.20, str(len(lc_df.loc[t_sup:])), fontsize=15, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        ax.axvline(t_inf)
        ax.axvline(t_0, linestyle='--')
        ax.axvline(t_sup)


    def plot_sql_available(ax, sql_lc_df, t_inf, t_sup):
        for mjd in list(sql_lc_df.index):
            ax.axvline(mjd, ymin=0.02, ymax=0.09, linestyle='solid', color='grey')


    def plot_lightcurve(ax, zfilter):
        if lc_dict[zfilter]:
            lc_dict[zfilter]['lc']['flux'].plot(ax=ax, yerr=lc_dict[zfilter]['lc']['flux_err'], linestyle='None', marker='.', color='blue')
            plot_obs_count(ax, lc_dict[zfilter]['lc'], t_0, t_inf, t_sup)
            ax.grid(ls='--', linewidth=0.8)
            ax.set_xlabel("MJD")
            ax.set_ylabel("Flux - {}".format(zfilter))

            if lc_dict[zfilter]['sql_lc'] is not None:
                plot_sql_available(ax, lc_dict[zfilter]['sql_lc'], t_inf, t_sup)

        else:
            ax.text(0.5, 0.5, "No data", fontsize=30, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')



    with plt.style.context('seaborn-whitegrid'):

        fig, ax = plt.subplots(figsize=(15, 8), nrows=3, ncols=1, sharex=True)
        fig.suptitle("{}".format(ztfname))

        ax = plt.subplot(3, 1, 1)
        plot_lightcurve(ax, 'zg')

        ax = plt.subplot(3, 1, 2)
        plot_lightcurve(ax, 'zr')

        ax = plt.subplot(3, 1, 3)
        plot_lightcurve(ax, 'zi')

        plt.tight_layout()

        if plot:
            plt.show()
        else:
            plt.savefig(save_folder.joinpath("{}/{}".format(save_folder, ztfname)).with_suffix(".png"), dpi=300)
            plt.savefig(save_folder.joinpath("{}/{}".format(save_folder, ztfname)).with_suffix(".pdf"), dpi=300)

        plt.close()

    def _fix_ipac_file(filename):
        split = filename.split(".")
        return "{}_{}.fits".format(split[0], split[1])


    def generate_lc_df(lc_dict, zfilter, t_min, t_max):
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

        _save_df_filter(df_lc_zg, df_params_zg, 'zg', first=True)
        _save_df_filter(df_lc_zr, df_params_zr, 'zr')
        _save_df_filter(df_lc_zi, df_params_zi, 'zi')


    if not plot:
        save_df(generate_lc_df(lc_dict, 'zg', t_min, t_max),
                generate_lc_df(lc_dict, 'zr', t_min, t_max),
                generate_lc_df(lc_dict, 'zi', t_min, t_max),
                generate_params_df(lc_dict, 'zg'),
                generate_params_df(lc_dict, 'zr'),
                generate_params_df(lc_dict, 'zi'))

        # fgallery description file
        with open(save_folder.joinpath("{}.txt".format(ztfname)), mode='w') as f:
            f.write("{}\n".format(ztfname))
            f.write("z={}\n".format(redshift_df.loc[ztfname]['redshift']))
            f.write("(ra, dec)=({}, {})".format(redshift_df.loc[ztfname]['host_ra'], redshift_df.loc[ztfname]['host_dec']))

        print(".", end="", flush=True)



Parallel(n_jobs=n_jobs)(delayed(estimate_lc_params)(ztfname) for ztfname in ztfnames)

print("")
