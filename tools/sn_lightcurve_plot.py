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

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs import utils
import matplotlib
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, wait, get_worker
from dask_jobqueue import SLURMCluster, SGECluster
import ztfquery.io
from ztfimg.science import ScienceQuadrant
from imageproc.composable_functions import BiPol2D
from ztfimg.stamps import stamp_it
import numpy as np
import pyloka

import list_format


filtercodes = ['zg', 'zr', 'zi']
filtercode_colors = {'zg': 'green', 'zr': 'red', 'zi': 'orange'}
idx_to_marker = {0: "1", 1: "2", 2: "3", 3: "4", 4: "v", 5: "^", 6: "<", 7: ">"}
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


def poly2d_from_file(filename):
    with open(filename, 'r') as f:
        f.readline()
        deg_str = f.readline()[:-1]
        degree = int(deg_str.split(" ")[1])
        coeff_str = " ".join(f.readline()[:-1].split())
        coeffs = list(map(float, coeff_str.split(" ")))

    coeffs_1 = coeffs[:int(len(coeffs)/2)]
    coeffs_2 = coeffs[int(len(coeffs)/2):]

    def _extract_coeffs(coeffs):
        idx = 0
        c = np.zeros([degree, degree])
        for d in range(degree):
            p, q = d, 0
            while p >= 0:
                c[p, q] = coeffs[idx]
                idx += 1
                p -= 1
                q += 1

        return c

    c_1 = _extract_coeffs(coeffs_1)
    c_2 = _extract_coeffs(coeffs_2)

    def _apply_pol(x, y):
        return np.stack([np.polynomial.polynomial.polyval2d(x, y, c_1),
                         np.polynomial.polynomial.polyval2d(x, y, c_2)])

    return _apply_pol


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--lc-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--stamp-size', type=int, default=32)
    argparser.add_argument('--output', type=pathlib.Path, required=True)

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


    for ztfname in ztfnames:
        def _get_lc_info(filtercode):
            sn_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not sn_folder.exists():
                return

            smphot_lc_sn_file = sn_folder.joinpath("smphot_output/lightcurve_sn.dat")
            smphot_lc_fit_file = sn_folder.joinpath("smphot_output/lc2fit.dat")

            if not smphot_lc_sn_file.exists() or not smphot_lc_fit_file.exists():
                return

            lc_info = {}

            with open(smphot_lc_sn_file, 'r') as f:
                _, sn_flux_df = list_format.read_list(f)

            with open(smphot_lc_fit_file, 'r') as f:
                globals_fit_df, fit_df = list_format.read_list(f)

            sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

            ref_exp = globals_fit_df['referenceimage']
            lc_info['ref_exp'] = ref_exp

            lc_info['sn_flux'] = sn_flux_df
            lc_info['sn_flux']['fieldid'] = fit_df['name'].apply(lambda x: int(x.split("_")[2]))
            lc_info['fieldids'] = tuple(set(lc_info['sn_flux']['fieldid']))

            with fits.open(sn_folder.joinpath("{}/calibrated.fits".format(ref_exp))) as hdul:
                w = WCS(hdul[0].header)

            init_skycoord = SkyCoord(sn_parameters['sn_ra'].item(), sn_parameters['sn_dec'].item(), unit='deg')
            init_px = (init_skycoord.to_pixel(w)[0].item(), init_skycoord.to_pixel(w)[1].item())

            lc_info['init_radec'] = init_skycoord
            lc_info['init_px'] = init_px

            fit_px = (sn_flux_df['x'].iloc[0], sn_flux_df['y'].iloc[0])
            #fit_skycoord = utils.pixel_to_skycoord(*fit_px, w)
            fit_skycoord = SkyCoord.from_pixel(*fit_px, w)
            fit_ra, fit_dec = fit_skycoord.ra, fit_skycoord.dec
            loka_radec = pyloka.pix2radec(str(sn_folder.joinpath("{}/calibrated.fits".format(ref_exp))), [fit_px[0]], [fit_px[1]])

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

            pol = poly2d_from_file(sn_folder.joinpath("pmfit/transfoTo{}.dat".format(lc_info['t0_exp_file'])))
            fit_px_t0 = pol(*fit_px)
            init_px_t0 = pol(*init_px)

            # Close view of the SN
            with fits.open(sn_folder.joinpath("smphot_output/{}.fits".format(lc_info['t0_exp_file']))) as hdul:
                lc_info['t0_sn_stamp'] = hdul[0].data

            z = ScienceQuadrant.from_filename(lc_info['t0_exp_file']+"_sciimg.fits")
            t0_quadrant = z.get_dataclean()

            sn_host_stamp = np.array(stamp_it(t0_quadrant, fit_px_t0[0], fit_px_t0[1], args.stamp_size, asarray=True))
            lc_info['t0_exp_host_stamp'] = sn_host_stamp

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


        def _plot_lc_info(lc_info, i, first):
            plt.subplot(3, 3, i*3+1)
            plt.xlabel("$x$ [pixel]")
            plt.ylabel("$y$ [pixel]")
            if first:
                plt.title("Host galaxy")
            plt.imshow(np.log(np.fmax(lc_info['t0_exp_host_stamp'], 1.)), cmap='gray', origin='upper',
                       extent=[lc_info['init_px'][0]-args.stamp_size/2,
                               lc_info['init_px'][0]+args.stamp_size/2,
                               lc_info['init_px'][1]-args.stamp_size/2,
                               lc_info['init_px'][1]+args.stamp_size/2])
            plt.plot(lc_info['fit_px'][0], lc_info['fit_px'][1], '.')

            plt.subplot(3, 3, i*3+2)
            plt.xlabel("$x$ [pixel]")
            plt.ylabel("$y$ [pixel]")
            if first:
                plt.title("Close view")
            plt.imshow(lc_info['t0_sn_stamp'], cmap='gray')

            plt.subplot(3, 3, i*3+3)
            for i, fieldid in enumerate(lc_info['fieldids']):
                sn_flux = lc_info['sn_flux'][lc_info['sn_flux']['fieldid'] == fieldid]
                plt.errorbar(sn_flux['mjd'], sn_flux['flux'], yerr=sn_flux['varflux'], color='black', ms=5., lw=0., marker=idx_to_marker[i], ls='', label=str(fieldid), elinewidth=1.)

            #plt.errorbar(lc_info['sn_flux']['mjd'], lc_info['sn_flux']['flux'], yerr=lc_info['sn_flux']['varflux'], color='black', marker='.', ls='')

            plt.xlim([lc_info['t_inf'], lc_info['t_sup']])
            plt.axvline(lc_info['t0'], color='black')
            plt.xlabel("MJD")
            plt.ylabel("Flux")
            plt.legend(title="Field ID")
            if first:
                plt.title("Lightcurve")
            plt.grid()


        def _plot_fitted_pos(lc_infos):
            # Get first lc_info available to get initial position
            first_lc_info = [lc_infos[filtercode] for filtercode in filtercodes if lc_infos[filtercode]][0]
            init_radec = np.array([first_lc_info['init_radec'].frame.data.lon.value, first_lc_info['init_radec'].frame.data.lat.value])
            plt.errorbar(init_radec[0], init_radec[1], marker='X', label='init', lw=0.)

            pos_list = [init_radec]
            err_list = []
            for filtercode in filtercodes:
                lc_info = lc_infos[filtercode]
                pos_radec = np.array([lc_info['fit_radec'].frame.data.lon.value, lc_info['fit_radec'].frame.data.lat.value])
                err_list.append(3600.*np.max(lc_info['fit_radec_var']))
                pos_list.append(pos_radec)
                plt.errorbar(pos_radec[0], pos_radec[1], marker='.', color=filtercode_colors[filtercode],
                             xerr=3600.*lc_info['fit_radec_var'][0], yerr=3600.*lc_info['fit_radec_var'][1], label=filtercode, lw=0., elinewidth=1.)

            pos = np.stack(pos_list)
            off = 0.5*np.max(err_list)
            plt.xlim([np.min(pos[:, 0]) - off, np.max(pos[:, 0]) + off])
            plt.ylim([np.min(pos[:, 1]) - off, np.max(pos[:, 1]) + off])
            # plt.axis('equal')
            plt.xlabel("$x$ [deg]")
            plt.ylabel("$y$ [deg]")
            plt.grid()
            plt.legend()
            plt.title("Fitted position per band")

        # Do plots
        plt.subplots(ncols=3, nrows=3, constrained_layout=True, figsize=(15., 9.), gridspec_kw={'width_ratios': [1., 1., 3.], 'height_ratios': [1., 1., 1.]})
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
        plt.savefig(args.output.joinpath("{}_pos.png".format(ztfname)), dpi=200.)
        plt.close()
