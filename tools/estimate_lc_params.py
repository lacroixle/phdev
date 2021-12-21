#!/usr/bin/env python3
import logging
import sys
import os
import pathlib
import distutils.util
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from joblib import Parallel, delayed
from ztfquery import query
import astropy.time

#ZTF19aaripqw

zfilters = ['zr', 'zg', 'zi']
zfilter_plot_color = dict(zip(zfilters, ['blue', 'red', 'orange']))

argparser = argparse.ArgumentParser(description="Lightcurve estimation tools.")
argparser.add_argument("--output", type=pathlib.Path, help="Output folder")
argparser.add_argument('-j', dest='n_jobs', type=int, default=1, help="Number of jobs to launch.")
argparser.add_argument('--ztfname', type=str, nargs='?', help="Process a specific SN 1a.")
argparser.add_argument('-v', type=int, dest='verbosity', default=0, help="Verbosity level.")
argparser.add_argument('--cosmodr', type=pathlib.Path, help="Cosmo DR folder.")
argparser.add_argument('--off_mul', type=int, default=3, help="Off SN 1a image statistics multiplier.")
argparser.add_argument('--plot', action='store_true', help="If set, will plot the lightcurve. Only when --ztfname is set.")
argparser.add_argument('--zmax', type=float, default=5., help="")
argparser.add_argument('--sql-request', dest='sql_request', action='store_true')

args = argparser.parse_args()

verbosity = args.verbosity
n_jobs = args.n_jobs
t0_inf = 50
t0_sup = 120
plot = args.plot
zmax = args.zmax
off_mul = args.off_mul
sql_request = args.sql_request
output_folder = args.output.expanduser().resolve()

if args.cosmodr:
    cosmo_dr_folder = args.cosmodr
elif 'COSMO_DR_FOLDER' in os.environ.keys():
    cosmo_dr_folder = pathlib.Path(os.environ.get("COSMO_DR_FOLDER"))
else:
    print("Cosmo DR folder not set! Either set COSMO_DR_FOLDER environnement variable or use the --cosmodr parameter.")
    exit(-1)

salt_df = pd.read_csv(cosmo_dr_folder.joinpath("params/DR2_SALT2fit_params.csv"), delimiter=",", index_col="name")
redshift_df = pd.read_csv(cosmo_dr_folder.joinpath("params/DR2_redshifts.csv"), delimiter=",", index_col="ztfname")
lightcurve_folder = cosmo_dr_folder.joinpath("lightcurves/").expanduser().resolve()


ztfnames = []
if args.ztfname:
    ztfnames = [args.ztfname]
    n_jobs = 1
else:
    ztf_files = lightcurve_folder.glob("*.csv")

    ztfnames = [ztf_file.stem.split("_")[0] for ztf_file in ztf_files]

# For some reason this sn does not exist in the SALT db
blacklist = ["ZTF18aaajrso"]


def estimate_lc_params(ztfname):
    if ztfname in blacklist:
        return

    def extract_interval(ztfname, t0_inf, t0_sup, off_mul, do_sql_request=True):
        lc_df = pd.read_csv(lightcurve_folder.joinpath("{}_LC.csv".format(ztfname)), delimiter="\s+", index_col="mjd")
        lc_df = lc_df[(np.abs(stats.zscore(lc_df['flux_err'])) < zmax)]

        sql_lc_df = None
        if sql_request:
            def _sql_request():
                zquery = query.ZTFQuery()
                zquery.load_metadata(radec=(redshift_df.loc[ztfname]['host_ra'], redshift_df.loc[ztfname]['host_dec']))
                sql_lc_df = zquery.metatable

                if verbosity >= 1:
                    print("{}: found SQL {} entries".format(ztfname, len(sql_lc_df)))

                return sql_lc_df

            sql_successfull = False
            max_sql_attempts = 5
            for i in range(0, max_sql_attempts):
                if sql_successfull:
                    break

                if i > 1:
                    print("{}: SQL attempt n°{}".format(ztfname, i))

                sql_lc_df = _sql_request()
                if 'obsmjd' in sql_lc_df.columns:
                    sql_successfull = True

            if not sql_successfull:
                print("{}: no obsjd column in metatable after {} attempts".format(ztfname, max_sql_attempts))
                return



            # Add an obsmjd column and set it as index
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

            return_dict = {'lc': lc_f_df.loc[t_min:t_max], 't_min': t_min, 't_max': t_max}

            if sql_request:
                return_dict['sql_lc'] = sql_lc_df.loc[sql_lc_df['filtercode'] == filt]

            return return_dict


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
            lc_dict[zfilter]['lc']['flux'].plot(ax=ax, yerr=lc_dict[zfilter]['lc']['flux_err'], linestyle='None', marker='.', color=zfilter_plot_color[zfilter])
            plot_obs_count(ax, lc_dict[zfilter]['lc'], t_0, t_inf, t_sup)
            ax.grid(ls='--', linewidth=0.8)
            ax.set_xlabel("MJD")
            ax.set_ylabel("Flux - {}".format(zfilter))

            if 'sql_lc' in lc_dict[zfilter].keys():
                plot_sql_available(ax, lc_dict[zfilter]['sql_lc'], t_inf, t_sup)

        else:
            ax.text(0.5, 0.5, "No data", fontsize=30, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')


    with plt.style.context('seaborn-whitegrid'):

        fig, ax = plt.subplots(figsize=(15, 8), nrows=3, ncols=1, sharex=True)
        fig.suptitle("{}".format(ztfname))

        [plot_lightcurve(plt.subplot(3, 1, i+1), zfilter) for i, zfilter in enumerate(zfilters)]

        plt.tight_layout()

        if plot:
            plt.show()
        else:
            plt.savefig(output_folder.joinpath("{}/{}".format(output_folder, ztfname)).with_suffix(".png"), dpi=300)
            plt.savefig(output_folder.joinpath("{}/{}".format(output_folder, ztfname)).with_suffix(".pdf"), dpi=300)

        plt.close()

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

                df_lc.to_csv(output_folder.joinpath("{}_{}.csv".format(ztfname, zfilter)), sep=",")
                df_params.to_csv(output_folder.joinpath("{}_{}_params.csv".format(ztfname, zfilter)), sep=",")
                df_lc.to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter), mode=mode)
                df_params.to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='params_{}'.format(zfilter))

        _save_df_filter(df_lc_zg, df_params_zg, 'zg', first=True)
        _save_df_filter(df_lc_zr, df_params_zr, 'zr')
        _save_df_filter(df_lc_zi, df_params_zi, 'zi')


    if output_folder:
        save_df(generate_lc_df(lc_dict, 'zg'),
                generate_lc_df(lc_dict, 'zr'),
                generate_lc_df(lc_dict, 'zi'),
                generate_params_df(lc_dict, 'zg'),
                generate_params_df(lc_dict, 'zr'),
                generate_params_df(lc_dict, 'zi'))

        # fgallery description file
        with open(output_folder.joinpath("{}.txt".format(ztfname)), mode='w') as f:
            f.write("{}\n".format(ztfname))
            f.write("z={}\n".format(redshift_df.loc[ztfname]['redshift']))
            f.write("(ra, dec)=({}, {})".format(redshift_df.loc[ztfname]['host_ra'], redshift_df.loc[ztfname]['host_dec']))

        if len(ztfnames) > 1 and verbosity == 0:
            print(".", end="", flush=True)



Parallel(n_jobs=n_jobs)(delayed(estimate_lc_params)(ztfname) for ztfname in ztfnames)

if not plot and len(ztfnames) > 1:
    print("")
