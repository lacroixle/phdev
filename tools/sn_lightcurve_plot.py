#!/usr/bin/env python3


import argparse
import pathlib
import subprocess
import logging
import datetime
import os
import time
import shutil
import sys
import socket
import copy
import traceback
import gc

from scipy import stats
from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs import utils
import matplotlib
import matplotlib.pyplot as plt
# import dask
# from dask import delayed, compute
# from dask.distributed import Client, LocalCluster, wait, get_worker
# from dask_jobqueue import SLURMCluster, SGECluster
import ztfquery.io
from ztfimg.science import ScienceQuadrant
import numpy as np
#import pyloka
import utils
from scipy.interpolate import LSQUnivariateSpline
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

filtercodes = ['zg', 'zr', 'zi']
filtercode_colors = {'zg': 'green', 'zr': 'red', 'zi': 'orange'}
#idx_to_marker = {0: "1", 1: "2", 2: "3", 3: "4", 4: "v", 5: "^", 6: "<", 7: ">"}
idx_to_marker = {0: "^", 1: "v", 2: "<", 3: ">", 4: "1", 5: "2", 6: "3", 7: "4"}


def radec_covar(px, varpx, w, h=1e-4):
    skycoord_ra_plus = SkyCoord.from_pixel(px[0]+h, px[1], w)
    skycoord_ra_minus = SkyCoord.from_pixel(px[0]-h, px[1], w)
    skycoord_dec_plus = SkyCoord.from_pixel(px[0], px[1]+h, w)
    skycoord_dec_minus = SkyCoord.from_pixel(px[0], px[1]-h, w)

    def _to_ndarray(x):
        return np.array([x.frame.data.lon.value, x.frame.data.lat.value])

    ra_plus = _to_ndarray(skycoord_ra_plus)
    ra_minus = _to_ndarray(skycoord_ra_minus)
    dec_plus = _to_ndarray(skycoord_dec_plus)
    dec_minus = _to_ndarray(skycoord_dec_minus)

    d_ra = (ra_plus - ra_minus)/(2*h)
    d_dec = (dec_plus - dec_minus)/(2*h)

    P = np.stack([d_ra, d_dec])
    C = np.diag(varpx)

    return tuple(np.diag(P@C@P.T).tolist())


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--lc-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--stamp-size', type=int, default=32)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('--no-fp', action='store_true', help="Set to disable forced photometry comparison (e.g. when not available).")
    argparser.add_argument('--mag', action='store_true', help="If set, plot lightcurve in mag unit, else in ADU.")
    argparser.add_argument('-j', type=int, default=1)

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()
    args.lc_folder = args.lc_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    ztfnames = None
    if args.ztfname is not None:
        if args.ztfname.stem == str(args.ztfname):
            ztfnames = [str(args.ztfname)]
        else:
            args.ztfname = args.ztfname.expanduser().resolve()
            if args.ztfname.exists():
                with open(args.ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
            else:
                pass

    print("Found {} SN1a".format(len(ztfnames)))


    def plot_ztf_lightcurve(ztfname):
        def _get_lc_info(filtercode):
            sn_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not sn_folder.exists():
                return

            smphot_lc_sn_file = sn_folder.joinpath("smphot/lightcurve_sn.dat")
            smphot_lc_fit_file = sn_folder.joinpath("smphot/lc2fit.dat")

            if not smphot_lc_sn_file.exists() or not smphot_lc_fit_file.exists():
                return

            lc_info = {}

            with open(smphot_lc_sn_file, 'r') as f:
                _, sn_flux_df = utils.read_list(f)

            with open(smphot_lc_fit_file, 'r') as f:
                globals_fit_df, fit_df = utils.read_list(f)

            nans = sn_flux_df.isna().any(axis=1)
            sn_flux_df = sn_flux_df.loc[~nans]
            fit_df = fit_df.loc[~nans]

            if len(sn_flux_df) == 0:
                return

            sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

            ref_exp = globals_fit_df['referenceimage']
            lc_info['ref_exp'] = ref_exp
            lc_info['filtercode'] = filtercode

            lc_info['sn_flux'] = sn_flux_df
            lc_info['sn_flux']['fieldid'] = fit_df['name'].apply(lambda x: int(x.split("_")[2]))
            lc_info['fieldids'] = tuple(set(lc_info['sn_flux']['fieldid']))
            lc_info['sn_flux'] = lc_info['sn_flux']

            t = np.linspace(sn_flux_df['mjd'].min()+1., sn_flux_df['mjd'].max()-1., int(len(sn_flux_df)/3.))
            #lc_info['spline'] = LSQUnivariateSpline(sn_flux_df['mjd'].to_numpy(), sn_flux_df['flux'].to_numpy(), t)

            ref_exp_path = ztfquery.io.get_file(ref_exp + "_sciimg.fits", downloadit=False)
            #with fits.open(sn_folder.joinpath("{}/calibrated.fits".format(ref_exp))) as hdul:
            with fits.open(ref_exp_path) as hdul:
                w = WCS(hdul[0].header)

            init_skycoord = SkyCoord(sn_parameters['sn_ra'].item(), sn_parameters['sn_dec'].item(), unit='deg')
            init_px = np.array([init_skycoord.to_pixel(w)[0].item(), init_skycoord.to_pixel(w)[1]])

            lc_info['init_radec'] = init_skycoord
            lc_info['init_px'] = init_px

            fit_px = np.array([sn_flux_df['x'].iloc[0], sn_flux_df['y'].iloc[0]])

            fit_skycoord = SkyCoord.from_pixel(*fit_px, w)
            fit_ra, fit_dec = fit_skycoord.ra, fit_skycoord.dec
            #loka_radec = pyloka.pix2radec(str(sn_folder.joinpath("{}/calibrated.fits".format(ref_exp))), [fit_px[0]], [fit_px[1]])

            lc_info['fit_px'] = fit_px
            lc_info['fit_radec'] = fit_skycoord

            lc_info['fit_px_var'] = (sn_flux_df['varx'].iloc[0], sn_flux_df['vary'].iloc[0])

            lc_info['fit_radec_var'] = radec_covar(fit_px, lc_info['fit_px_var'], w)

            # Get stamp of SN at t0
            t0 = sn_parameters['t0mjd'].item()
            t0_idx = np.argmin(np.abs(fit_df['Date']-t0))

            lc_info['t0'] = t0
            lc_info['t_inf'] = sn_parameters['t_inf'].values
            lc_info['t_sup'] = sn_parameters['t_sup'].values

            lc_info['t0_exp'] = fit_df.iloc[t0_idx]['Date']
            lc_info['t0_exp_file'] = fit_df.iloc[t0_idx]['name']


            # Close view of the SN
            with fits.open(sn_folder.joinpath("smphot/{}.fits".format(lc_info['t0_exp_file']))) as hdul:
                lc_info['t0_sn_stamp'] = hdul[0].data

            # z = ScienceQuadrant.from_filename(lc_info['t0_exp_file']+"_sciimg.fits", use_dask=False)
            # t0_quadrant = z.get_dataclean()

            # sn_host_stamp = np.array(stamp_it(t0_quadrant, fit_px_t0[0], fit_px_t0[1], args.stamp_size, asarray=True))
            # lc_info['t0_exp_host_stamp'] = sn_host_stamp

            # Get forced photometry lightcurve if ordered
            if not args.no_fp:
                lc_info['lc_fp'] = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='lc_fp_{}'.format(filtercode))
                lc_info['lc_fp'] = lc_info['lc_fp'].loc[lc_info['lc_fp'].index.to_series().between(lc_info['t_inf'][0], lc_info['t_sup'][0])]

            return lc_info


        ztffolder = args.wd.joinpath(ztfname)
        lc_infos = {}
        # First get all relevant informations
        for filtercode in filtercodes:
            if ztffolder.joinpath(filtercode).exists():
                lc_band_info = _get_lc_info(filtercode)

                if lc_band_info is not None:
                    lc_infos[filtercode] = lc_band_info

        if not lc_infos:
            print("Found no data for {}... Skipping.".format(ztfname))
            return
        else:
            print(ztfname)


        def _plot_lc_info(lc_info, i, first):
            plt.subplot(3, 2, i*2+1)
            plt.xlabel("$x$ [pixel]")
            plt.ylabel("$y$ [pixel] - {}".format(lc_info['filtercode']))
            # if first:
            #     plt.title("Host galaxy")
            # plt.imshow(np.log(np.fmax(lc_info['t0_exp_host_stamp'], 1.)), cmap='gray', origin='upper',
            #            extent=[lc_info['init_px'][0]-args.stamp_size/2,
            #                    lc_info['init_px'][0]+args.stamp_size/2,
            #                    lc_info['init_px'][1]-args.stamp_size/2,
            #                    lc_info['init_px'][1]+args.stamp_size/2])
            # plt.plot(lc_info['fit_px'][0], lc_info['fit_px'][1], '.')

            # plt.subplot(3, 3, i*3+2)
            # plt.xlabel("$x$ [pixel]")
            # plt.ylabel("$y$ [pixel]")
            if first:
                plt.title("Close view")
            plt.imshow(lc_info['t0_sn_stamp'], cmap='gray')

            plt.subplot(3, 2, i*2+2)
            for j, fieldid in enumerate(lc_info['fieldids']):
                sn_flux = lc_info['sn_flux'][lc_info['sn_flux']['fieldid'] == fieldid]

                if len(sn_flux) == 0:
                    continue

                if args.mag:
                    sn_flux['flux'] = -2.5*np.log10(sn_flux['flux']+sn_flux['flux'].min())

                plt.errorbar(sn_flux['mjd'].to_numpy(), sn_flux['flux'].to_numpy(), yerr=sn_flux['varflux'].to_numpy(), color='black', ms=5., lw=0., marker=idx_to_marker[j], ls='', label=str(fieldid), elinewidth=1.)

            #t = np.linspace(lc_info['sn_flux']['mjd'].min(), lc_info['sn_flux']['mjd'].max(), 500)
            #plt.plot(t, lc_info['spline'](t), color='grey')
            plt.xlim([lc_info['t_inf'], lc_info['t_sup']])
            plt.axvline(lc_info['t0'], color='black')
            plt.xlabel("MJD")
            if args.mag:
                plt.ylabel("$m$")
            else:
                plt.ylabel("Flux [ADU]")

            plt.legend(title="Field ID")
            if first:
                plt.title("Lightcurve")
            plt.grid()


        def _plot_fitted_pos(lc_infos):
            init_px = np.array([0., 0.])

            plt.errorbar(init_px[0], init_px[1], marker='X', label='init', lw=0.)
            pos_list = [init_px]

            err_list = []
            #for filtercode in filtercodes:
            for filtercode in lc_infos.keys():
                lc_info = lc_infos[filtercode]
                pos_px = lc_info['fit_px'] - lc_info['init_px']
                err_list.append(np.max(lc_info['fit_px_var']))
                pos_list.append(pos_px)
                plt.errorbar(pos_px[0], pos_px[1], marker='.', color=filtercode_colors[filtercode],
                             xerr=lc_info['fit_px_var'][0], yerr=lc_info['fit_px_var'][1], label=filtercode, lw=0., elinewidth=1.)

            pos = np.stack(pos_list)
            off = np.max(err_list) + 0.1
            plt.xlim([np.min(pos[:, 0]) - off, np.max(pos[:, 0]) + off])
            plt.ylim([np.min(pos[:, 1]) - off, np.max(pos[:, 1]) + off])
            plt.axis('equal')
            plt.xlabel("$x$ [pixel]")
            plt.ylabel("$y$ [pixel]")
            plt.grid()
            plt.legend()

        def _plot_fp_diff(lc_info, i, first, columns=False):
            if columns:
                plt.subplot(2, 1, 1)
            else:
                plt.subplot(3, 2, i*2 + 1)

            if first and not columns:
                plt.title("Forced photometry lightcurve")

            for j, fieldid in enumerate(lc_info['fieldids']):
                sn_flux = lc_info['lc_fp'][lc_info['lc_fp']['field_id'] == fieldid]
                to_plot = ~np.any([~(np.abs(stats.zscore(sn_flux['flux_err'])) < 2.), ~(np.abs(stats.zscore(sn_flux['flux'])) < 5)], axis=0)
                #print("To plot: {} ({} removed)".format(sum(to_plot), len(sn_flux)-sum(to_plot)))
                plt.errorbar(sn_flux.index[to_plot].to_numpy(), sn_flux['flux'][to_plot].to_numpy(), yerr=sn_flux['flux_err'][to_plot].to_numpy(), color='black', ms=5., lw=0., marker=idx_to_marker[j], ls='', label=str(fieldid), elinewidth=1.)

            plt.legend(title="Field ID")
            plt.grid()

            if not columns:
                plt.xlabel("MJD")
                plt.axvline(lc_info['t0'], color='black')

            if columns:
                plt.ylabel("FP flux [ADU]")

            plt.xlim(lc_info['t_inf'], lc_info['t_sup'])

            if columns:
                plt.subplot(2, 1, 2)
            else:
                plt.subplot(3, 2, i*2 + 2)

            if first and not columns:
                plt.title("Scene modeling lightcurve")

            for j, fieldid in enumerate(lc_info['fieldids']):
                sn_flux = lc_info['sn_flux'][lc_info['sn_flux']['fieldid'] == fieldid]
                plt.errorbar(sn_flux['mjd'].to_numpy(), sn_flux['flux'].to_numpy(), yerr=sn_flux['varflux'].to_numpy(), color='black', ms=5., lw=0., marker=idx_to_marker[j], ls='', label=str(fieldid), elinewidth=1.)

            if not columns:
                plt.legend(title="Field ID")
                plt.axvline(lc_info['t0'], color='black')

            plt.grid()
            plt.xlabel("MJD")
            if not columns:
                plt.ylabel("Flux - {}".format(lc_info['filtercode']))
            else:
                plt.ylabel("SMP flux [ADU]")
            plt.xlim(lc_info['t_inf'], lc_info['t_sup'])

        # Do plots
        plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(15., 9.), gridspec_kw={'width_ratios': [1., 5.], 'height_ratios': [1., 1., 1.]})
        first = True
        for i, filtercode in enumerate(filtercodes):
            if filtercode in lc_infos:
                _plot_lc_info(lc_infos[filtercode], i, first)
                first = False
            else:
                ax = plt.subplot(3, 1, i+1)
                ax.text(0.5, 0.5, "No data", fontsize=30, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
                ax.axis('off')

        plt.savefig(args.output.joinpath("{}.png".format(ztfname)), dpi=200.)
        plt.close()

        plt.figure(figsize=(5., 5.), constrained_layout=True)
        _plot_fitted_pos(lc_infos)
        plt.title("{}-{} fitted positions".format(ztfname, filtercode))
        plt.savefig(args.output.joinpath("{}_pos.png".format(ztfname)), dpi=200.)
        plt.close()

        mode = 'w'
        for filtercode in lc_infos.keys():
            lc_infos[filtercode]['sn_flux'].to_hdf(args.output.joinpath("SMP_{}.hd5".format(ztfname)), key=filtercode, mode=mode)
            mode = 'a'

        if not args.no_fp:
            plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(15., 9.))
            first = True
            for i, filtercode in enumerate(filtercodes):
                if filtercode in lc_infos.keys():
                    _plot_fp_diff(lc_infos[filtercode], i, first)
                    # mode = 'a'
                    # if first:
                    #     mode = 'w'

                    # first = False

                    # Save
                else:
                    ax = plt.subplot(3, 1, i+1)
                    ax.text(0.5, 0.5, "No data", fontsize=30, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
                    ax.axis('off')

            plt.savefig(args.output.joinpath("{}_fp_diff.png".format(ztfname)), dpi=200.)
            plt.close()

            # Other version of the same plot, better suited for publications...
            for i, filtercode in enumerate(filtercodes):
                if filtercode in lc_infos.keys():
                    fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(8., 5.), gridspec_kw={'hspace': 0.})
                    for ax in axs:
                        ax.tick_params(which='both', direction='in')
                        ax.xaxis.set_minor_locator(AutoMinorLocator())
                        ax.yaxis.set_minor_locator(AutoMinorLocator())

                    plt.suptitle("{} - {} filter".format(ztfname, filtercode[1]))
                    _plot_fp_diff(lc_infos[filtercode], i, True, columns=True)
                    fig.align_ylabels(axs=axs)
                    plt.tight_layout()
                    plt.savefig(args.output.joinpath("{}_{}_fp_diff.pdf".format(ztfname, filtercode)))
                    plt.close()

        gc.collect()


    p = Parallel(n_jobs=args.j)(delayed(plot_ztf_lightcurve)(ztfname) for ztfname in ztfnames)

    # for ztfname in ztfnames:
    #     plot_ztf_lightcurve(ztfname)
    #     gc.collect()
