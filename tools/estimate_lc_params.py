#!/usr/bin/env python3
import logging
import sys
import os
import pathlib
import distutils.util
import argparse
import itertools
from gc import collect

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from joblib import Parallel, delayed
from ztfquery import query
from ztfquery import buildurl
import astropy.time
from ztfimg import io
import ztfquery.fields


zfilters = ['zg', 'zr', 'zi']
zfilter_plot_color = dict(zip(zfilters, ['green', 'red', 'orange']))

argparser = argparse.ArgumentParser(description="Lightcurve estimation tools.")
argparser.add_argument("--output", type=pathlib.Path, help="Output folder", required=True)
argparser.add_argument('-j', dest='n_jobs', type=int, default=1, help="Number of jobs to launch.")
argparser.add_argument('--ztfname', type=pathlib.Path, nargs='?', help="Process a specific (or a list of) SN 1a.")
argparser.add_argument('-v', type=int, dest='verbosity', default=0, help="Verbosity level.")
argparser.add_argument('--cosmodr', type=pathlib.Path, help="Cosmo DR folder.")
argparser.add_argument('--off-mul', dest='off_mul', type=int, default=3, help="Off SN 1a image statistics multiplier.")
argparser.add_argument('--plot', action='store_true', help="If set, will plot the lightcurve. Only when --ztfname is set.")

args = argparser.parse_args()

verbosity = args.verbosity
n_jobs = args.n_jobs
t0_inf = 50
t0_sup = 120
plot = args.plot
off_mul = args.off_mul
output_folder = args.output.expanduser().resolve()


if args.cosmodr:
    cosmo_dr_folder = args.cosmodr
elif 'COSMO_DR_FOLDER' in os.environ.keys():
    cosmo_dr_folder = pathlib.Path(os.environ.get("COSMO_DR_FOLDER"))
else:
    print("Cosmo DR folder not set! Either set COSMO_DR_FOLDER environnement variable or use the --cosmodr parameter.")
    exit(-1)


salt_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/archived/2021oct26/DR2_SALT2fit_params.csv"), delimiter=",", index_col="name")
redshift_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/archived/2021oct26/DR2_redshifts.csv"), delimiter=",", index_col="ztfname")
coords_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/archived/2021oct26/ztfdr2_coords.csv"), delimiter=" ", index_col="ztfname")

lightcurve_folder = cosmo_dr_folder.joinpath("lightcurves/previous_versions/2021oct26").expanduser().resolve()
# salt_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/ztfdr2_salt2_params.csv"))
# redshift_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/ztfdr2_redshifts.csv"))
# coords_df = pd.read_csv(cosmo_dr_folder.joinpath("tables/ztfdr2_coordinates.csv"))


ztfnames = []
if args.ztfname:
    if args.ztfname.stem == str(args.ztfname):
        ztfnames = [args.ztfname]
        n_jobs = 1
    else:
        args.ztfname = args.ztfname.expanduser().resolve()
        with open(args.ztfname, 'r') as f:
            ztfnames = [ztfname.strip() for ztfname in f.readlines() if not ztfname.strip()[0] == "#"]

else:
    ztf_files = lightcurve_folder.glob("*.csv")
    ztfnames = [ztf_file.stem.split("_")[0] for ztf_file in ztf_files]

ztfnames = list(map(str, ztfnames))


# For some reason this sn does not exist in the SALT db
blacklist = ["ZTF18aaajrso", "ZTF19aanxgle"]

empty_snae = []

def estimate_lc_params(ztfname):
    import warnings
    warnings.simplefilter("ignore")

    if ztfname in blacklist:
        return

    class EmptySQLResult(Exception):
        def __init__(self):
            pass

    def extract_interval(ztfname, t0_inf, t0_sup, off_mul, do_sql_request=True):
        # First load forced photometry lightcurve (useful for plotting)
        fp_lc_df = pd.read_csv(lightcurve_folder.joinpath("{}_LC.csv".format(ztfname)), delimiter="\s+", index_col="mjd")

        # Then retrieve available quadrants covering the host gallaxy position
        sql_lc_df = None
        def _sql_request():
            zquery = query.ZTFQuery()
            zquery.load_metadata(radec=(redshift_df.at[ztfname, 'host_ra'], redshift_df.at[ztfname, 'host_dec']), sql_query="infobits < 33554432 and field <= 1895")
            sql_lc_df = zquery.metatable

            if len(sql_lc_df) == 0:
                if verbosity >= 1:
                    print("{}: no SQL entry found - discarded".format(ztfname))

                empty_snae.append(ztfname)

                raise EmptySQLResult()

            if verbosity >= 1:
                print("{}: found {} SQL entries".format(ztfname, len(sql_lc_df)))

            return sql_lc_df

        # For some reason (especially when heavely multithreaded), the metatable gets corrupted
        # In that case we try again several times
        sql_successfull = False
        max_sql_attempts = 5
        for i in range(0, max_sql_attempts):
            if sql_successfull:
                break

            if i > 1:
                print("{}: SQL attempt n°{}".format(ztfname, i))

            sql_lc_df = _sql_request()
            if 'obsjd' in sql_lc_df.columns:
                sql_successfull = True

        if not sql_successfull:
            print("{}: no obsjd column in metatable after {} attempts".format(ztfname, max_sql_attempts))
            return

        # If no quadrant can be found, skip processing
        if len(sql_lc_df) == 0:
            return

        # Add an obsmjd column and set it as index
        sql_lc_df['obsmjd'] = sql_lc_df['obsjd'].apply(lambda jd: astropy.time.Time(jd, format='jd').mjd)
        sql_lc_df.set_index('obsmjd', inplace=True)
        sql_lc_df.sort_index(inplace=True)

        # Add reference IPAC/IRSA image name
        sql_lc_df['ipac_file'] = buildurl.build_filename_from_dataframe(sql_lc_df)

        # Fix optimisation bug (convert object types into strings)
        col_to_str = ['filtercode', 'imgtype', 'imgtypecode', 'obsdate', 'ipac_pub_date', 'ipac_file']
        sql_lc_df[col_to_str] = sql_lc_df[col_to_str].astype('str')

        # Used to check for calibration fields & for gaia calibrators
        # Thank you StackOverflow
        # https://stackoverflow.com/questions/2213923/removing-duplicates-from-a-list-of-lists
        field_rcid_pairs = list(map(list, set(map(tuple, sql_lc_df[['field', 'rcid']].values.tolist()))))
        fields = [pair[0] for pair in field_rcid_pairs]
        rcids = [pair[1] for pair in field_rcid_pairs]

        # Zero order SN event time interval
        t_0 = salt_df.loc[ztfname, "t0"]
        t_inf = t_0 - t0_inf
        t_sup = t_0 + t0_sup

        # Get gaia calibrators
        if output_folder:
            try:
                gaia_cal_df = io.GaiaCalibrators.fetch_data(rcids, fields).drop(labels=['ps1_id', 'sdssdr13_id'], axis=1)
            except Exception as e:
                print(e)
                gaia_cal_df = pd.DataFrame()
            else:
                gaia_cal_df.reset_index(inplace=True)
                gaia_cal_df.set_index('Source', inplace=True)
                gaia_cal_df.rename(columns={'level_1': 'field', 'level_0': 'rcid'}, inplace=True)

        sn_info = {}
        sn_info['ztfname'] = ztfname
        sn_info['sn_ra'] = float(coords_df.loc[ztfname]['sn_ra'])
        sn_info['sn_dec'] = coords_df.loc[ztfname]['sn_dec']
        sn_info['host_ra'] = coords_df.loc[ztfname]['host_ra']
        sn_info['host_dec'] = coords_df.loc[ztfname]['host_dec']
        sn_info['redshift'] = redshift_df.loc[ztfname]['redshift']
        sn_info['fieldid'] = list(set(sql_lc_df['field']))
        sn_info['t0mjd'] = t_0
        sn_info['peakmjd'] = t_0
        sn_info['t_inf'] = t_inf
        sn_info['t_sup'] = t_sup
        sn_info_df = pd.DataFrame([sn_info])

        # Compute the time interval covering the SN event with off acquisitions (ie, only the host galaxy)
        def _compute_min_max_interval(lc_df, t_inf, t_sup, filtercode):
            lc_f_df = lc_df.loc[lc_df['filtercode'] == filtercode]

            obs_count = len(lc_f_df.loc[t_inf:t_sup])

            if obs_count == 0:
                return None

            idx_min = max(0, int(len(lc_f_df[:t_inf]) - off_mul*obs_count))
            idx_max = min(len(lc_f_df), int(len(lc_f_df[:t_sup]) + off_mul*obs_count)) - 1

            t_min = lc_f_df.iloc[idx_min].name
            t_max = lc_f_df.iloc[idx_max].name

            return {'sql_lc': lc_f_df.loc[t_min:t_max],
                    'fp_lc': fp_lc_df.loc[fp_lc_df['filter'] == 'ztf{}'.format(filtercode[1])].loc[t_min:t_max],
                    't_min': t_min,
                    't_max': t_max}
            # return {'sql_lc': lc_f_df.loc[t_min:t_max],
            #         'fp_lc': None,
            #         't_min': t_min,
            #         't_max': t_max}

        return dict([(filtercode, _compute_min_max_interval(sql_lc_df, t_inf, t_sup, filtercode)) for filtercode in zfilters]), t_inf, t_sup, t_0, gaia_cal_df, sn_info_df


    try:
        lc_dict, t_inf, t_sup, t_0, gaia_cal_df, sn_info_df = extract_interval(ztfname, t0_inf, t0_sup, off_mul)
    except EmptySQLResult:
        return

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
            to_plot = np.all([np.abs(stats.zscore(lc_dict[zfilter]['fp_lc']['flux_err'])) < 3, np.abs(stats.zscore(lc_dict[zfilter]['fp_lc']['flux'])) < 4], axis=0)

            lc_dict[zfilter]['fp_lc']['flux'][to_plot].plot(ax=ax, yerr=lc_dict[zfilter]['fp_lc']['flux_err'][to_plot], linestyle='None', marker='.', color=zfilter_plot_color[zfilter])
            plot_obs_count(ax, lc_dict[zfilter]['sql_lc'], t_0, t_inf, t_sup)
            ax.grid(ls='--', linewidth=0.8)
            ax.set_xlabel("MJD")
            ax.set_ylabel("{} - Flux [ADU]".format(zfilter))

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

    def generate_params_df(lc_dict, zfilter):
        if lc_dict[zfilter] is not None:
            params = {'t_0': t_0,
                    't_inf': t_inf,
                    't_sup': t_sup,
                    't_min': lc_dict[zfilter]['t_min'],
                    't_max': lc_dict[zfilter]['t_max'],
                    'off_mul': off_mul}

            return pd.DataFrame.from_records([params])

    def save_df(df_lc, gaia_cal_df, sn_info_df):
        def _save_df_filter(zfilter, first=False):
            if df_lc[zfilter] is not None:
                if first:
                    mode = 'w'
                else:
                    mode = 'a'

                df_lc[zfilter]['sql_lc'].to_csv(output_folder.joinpath("{}_{}.csv".format(ztfname, zfilter)), sep=",")
                generate_params_df(df_lc, zfilter).to_csv(output_folder.joinpath("{}_{}_params.csv".format(ztfname, zfilter)), sep=",")
                df_lc[zfilter]['sql_lc'].to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter), mode=mode)
                generate_params_df(lc_dict, zfilter).to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='params_{}'.format(zfilter))

                if 'fp_lc' in df_lc[zfilter].keys():
                    df_lc[zfilter]['fp_lc'].to_csv(output_folder.joinpath("{}_{}_fp.csv".format(ztfname, zfilter)), sep=",")
                    df_lc[zfilter]['fp_lc'].to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='lc_fp_{}'.format(zfilter))

        _save_df_filter('zg', first=True)
        _save_df_filter('zr')
        _save_df_filter('zi')

        gaia_cal_df.to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
        sn_info_df.to_hdf(output_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    if output_folder:
        save_df(lc_dict, gaia_cal_df, sn_info_df)

        # fgallery description file
        with open(output_folder.joinpath("{}.txt".format(ztfname)), mode='w') as f:
            f.write("{}\n".format(ztfname))
            f.write("z={}\n".format(redshift_df.loc[ztfname]['redshift']))
            f.write("(ra, dec)=({}, {})".format(redshift_df.loc[ztfname]['host_ra'], redshift_df.loc[ztfname]['host_dec']))

        if len(ztfnames) > 1 and verbosity == 0:
            print(".", end="", flush=True)

    collect(0)



Parallel(n_jobs=n_jobs)(delayed(estimate_lc_params)(ztfname) for ztfname in ztfnames)

if len(ztfnames) > 1:
    print("Discarded SN1a")
    for sn in empty_snae:
        print(sn)

if not plot and len(ztfnames) > 1:
    print("")
