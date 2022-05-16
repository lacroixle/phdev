#!/usr/bin/env python3

import argparse
import pathlib
import os
import time
import itertools
import copy
import gc

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import pyloka
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial
from joblib import Parallel, delayed

import list_format
import utils


filtercodes = ['zg', 'zr', 'zi']

def residuals_quadrant(quadrant_path, reference_quadrant_path, gaia_stars, filtercode, output_path):
    #pd.options.mode.chained_assignment = None
    # Astrometric residuals of psfstars with Gaia stars
    quadrant_name = quadrant_path.name

    if not args.wd.joinpath("{}/{}/pmfit/transfoTo{}.dat".format(ztfname, filtercode, quadrant_name)).exists():
        print("Could not find transformation!")
        return

    with open(quadrant_path.joinpath("psfstars.list"), 'r') as f:
        _, stars = utils.read_list(f)

    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        wcs = WCS(hdul[0].header)
        seeing = hdul[0].header['seeing']

    # cosmic_count = utils.ListTable.from_filename(quadrant_path.joinpath("se.list")).header['cosmic_count']
    cosmic_count = 0

    aperse_stars_count = utils.get_cat_size(quadrant_path.joinpath("aperse.list"))
    standalone_stars_count = utils.get_cat_size(quadrant_path.joinpath("standalone_stars.list"))
    psf_stars_count = utils.get_cat_size(quadrant_path.joinpath("psfstars.list"))

    gaia_coords_radec = SkyCoord(gaia_stars['ra'], gaia_stars['dec'], unit='deg')
    gaia_mask = utils.contained_in_exposure(gaia_coords_radec, wcs, return_mask=True)

    gaia_stars = gaia_stars[gaia_mask]
    gaia_coords_radec = gaia_coords_radec[gaia_mask]

    x, y = gaia_coords_radec.to_pixel(wcs)
    #gaia_stars['x'] = x
    #gaia_stars['y'] = y
    gaia_stars.insert(0, 'y', y)
    gaia_stars.insert(0, 'x', x)

    # Translate the gaia stars catalogue into quadrant pixel space
    i = utils.match_pixel_space(gaia_stars[['x', 'y']], stars, radius=3.)

    #refstars = gaia_stars.loc[i[i>=0]].reset_index(drop=True)
    refstars = gaia_stars.iloc[i[i>=0]].reset_index(drop=True)
    refstars_pair = stars[i>=0].reset_index(drop=True)
    gaia_stars_count = len(refstars)

    wcs_residuals = refstars[['x', 'y']].sub(refstars_pair[['x', 'y']])

    plt.subplots(1, 2, figsize=(10., 5.))
    plt.suptitle("WCS with Gaia stars residuals")
    plt.subplot(1, 2, 1)
    #plt.hist(refstars['x'] - refstars_pair['x'], bins=100, histtype='step', color='black')
    plt.hist(wcs_residuals['x'], bins=50, histtype='step', color='black')
    plt.grid()
    plt.xlabel("Residual [pixel]")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)

    plt.xlabel("Residual [pixel]")
    plt.grid()

    plt.savefig(output_path.joinpath("{}_wcs_gaia_residuals.png".format(quadrant_name)), dpi=150.)
    plt.close()

    # Check color with Gaia
    # Fit n degree polynomial

    if filtercode == 'zg':
        band_mag = 'bp'
        plt.ylabel("$g-B_p$")
    elif filtercode == 'zr':
        band_mag = refstars['rp']
        band_mag = 'rp'
        plt.ylabel("$r-R_p$")
    else:
        band_mag = None

    if band_mag is not None:
        color_min, color_max = -0.5, 0.5
        color = (refstars['bp'] - refstars['rp']) - np.mean(refstars['bp'] - refstars['rp'])

        fit_mask = np.all([(color >= color_min), (color <= color_max)], axis=0)
        fit_color = color[fit_mask]

        fit_y = -2.5*np.log10(refstars_pair['flux'][fit_mask]) - refstars[band_mag][fit_mask]
        y = -2.5*np.log10(refstars_pair['flux']) - refstars[band_mag]

        poly, fit_result = Polynomial.fit(fit_color, fit_y, args.color_degree, full=True, domain=[color_min, color_max])
        plt.errorbar(color, y, color='black', yerr=refstars_pair['eflux']/refstars_pair['flux'], marker='.', ls='')

        plt.plot(*poly.linspace())
        plt.axvline(color_min)
        plt.axvline(color_max)

        plt.xlabel("$B_p-R_p - \\langle B_p-R_p \\rangle $")
        plt.grid()
        plt.savefig(output_path.joinpath("{}_gaia_color.png".format(quadrant_name)), dpi=150.)
        plt.close()

        c_intercept = poly.coef[0]
    else:
        c_intercept = float('nan')

    # Check photometric linearity
    with open(reference_quadrant_path.joinpath("psfstars.list"), 'r') as f:
        _, reference_stars = utils.read_list(f)

    try:
        pol = utils.poly2d_from_file(args.wd.joinpath("{}/{}/pmfit/transfoTo{}.dat".format(ztfname, filtercode, quadrant_name)))
    except:
        return

    reference_stars[['x', 'y']] = reference_stars[['x', 'y']].apply(lambda x: pol(x[0], x[1]), axis=1, raw=True)
    utils.cat_to_ds9regions(reference_stars.add(1.).to_records(index=False), output_path.joinpath("{}_ref_regions.reg".format(quadrant_name)), color='green')
    utils.cat_to_ds9regions(stars.add(1.).to_records(index=False), output_path.joinpath("{}_regions.reg".format(quadrant_name)), color='red')

    i = utils.match_pixel_space(reference_stars, stars, radius=10.)
    refstars = reference_stars.iloc[i[i>=0]]
    refstars_pair = stars.iloc[i>=0]
    match_stars_count = len(refstars)

    if match_stars_count > 1:
        # Do linear regression
        regression_result = linregress(refstars['flux'], refstars_pair['flux'])
        alpha = regression_result.slope
        intercept = regression_result.intercept
        rvalue = regression_result.rvalue
        min_flux = refstars['flux'].min()
        max_flux = refstars['flux'].max()

        _, ax = plt.subplots(figsize=(5., 5.))
        plt.title("Flux comparison with reference quadrant stars")
        plt.plot(refstars['flux'], refstars_pair['flux'], 'x', color='black')
        plt.plot([min_flux, max_flux], [alpha*min_flux+intercept, alpha*max_flux+intercept], color='black', ls='--')
        plt.xlabel("Reference quadrant flux")
        plt.ylabel("Quadrant flux")
        plt.text(0.25, 0.1, "$\\alpha={}$".format(alpha), transform=ax.transAxes, size=20.)
        plt.text(0.25, 0.04, "$r={}$".format(rvalue), transform=ax.transAxes, size=20.)
        plt.grid()

        plt.savefig(output_path.joinpath("{}_flux_comparison.png".format(quadrant_name)), dpi=150.)
        plt.close()
    else:
        alpha = float('nan')
        rvalue = float('nan')

    gc.collect(2)
    gc.collect(1)
    gc.collect(0)

    return [alpha, rvalue, gaia_stars_count, match_stars_count, aperse_stars_count, standalone_stars_count, psf_stars_count, seeing, cosmic_count, c_intercept]#, wcs_residuals

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('--color-degree', type=int, default=1)
    argparser.add_argument('-j', type=int, default=1)
    argparser.add_argument('--force', action='store_true', help="Redo computations from scratch.")

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()
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

    # to_remove = ["ZTF19aavmnpb", "ZTF20abzjads", "ZTF20abzvxyk"]
    # for ztfname in to_remove:
    #     ztfnames.remove(ztfname)

    print("Found {} SN1a".format(len(ztfnames)))

    for ztfname in ztfnames:
        print("For {}".format(ztfname))
        for filtercode in filtercodes:
            print("In filter band {}".format(filtercode))
            band_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not band_folder.exists():
                continue

            if band_folder.joinpath("stats.csv").exists() and not args.force:
                print("Already done, continuing.")
                continue

            # Get reference quadrant
            reference_quadrant_path = utils.get_ref_quadrant_from_driver(args.wd.joinpath("{}/{}/{}_driver_{}".format(ztfname, filtercode, ztfname, filtercode)))
            if not reference_quadrant_path:
                print("{}-{}: no reference quadrant found!".format(ztfname, filtercode))
                continue

            print("Reference exposure={}".format(reference_quadrant_path.name))
            gaia_stars = pd.DataFrame.from_records(np.load(band_folder.joinpath("gaia.npy")))

            quadrant_paths = [quadrant_path for quadrant_path in list(band_folder.glob("ztf_*")) if quadrant_path.is_dir() and quadrant_path.joinpath("psfstars.list").exists()]
            try:
                quadrant_paths.remove(reference_quadrant_path)
            except ValueError:
                pass
            quadrant_paths.insert(0, reference_quadrant_path)

            output_path = args.output.joinpath("{}/{}".format(ztfname, filtercode))
            os.makedirs(output_path, exist_ok=True)

            results = Parallel(n_jobs=args.j)(delayed(residuals_quadrant)(quadrant_path, reference_quadrant_path, copy.deepcopy(gaia_stars), filtercode, output_path) for quadrant_path in quadrant_paths)
            results_mask = [result is not None for result in results]
            results = list(itertools.compress(results, results_mask))

            if len(results) == 0:
                print("No computation done for this band... continuing.")
                continue

            quadrant_paths = list(itertools.compress(quadrant_paths, results_mask))
            alphas = [result for result in results]
            stats_df = pd.DataFrame(data=np.array(alphas), index=list(map(lambda x: x.name, quadrant_paths)), columns=['alpha', 'r', 'n_gaia', 'n_match', 'n_aperse', 'n_standalone', 'n_psf', 'seeing', 'n_cosmic', 'color_intercept'])
            stats_df.index.name = 'quadrant'

            #wcs_residuals_df = pd.concat(wcs_residuals, ignore_index=True)


            # Now compute phtometric ratios
            c_intercept_ref = stats_df.at[reference_quadrant_path.name, 'color_intercept']
            alpha_df = 10**(-0.4*(stats_df['color_intercept']-c_intercept_ref))
            stats_df['alpha_gaia'] = alpha_df

            stats_df.to_csv(band_folder.joinpath("stats.csv"), sep=",")

            plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))

            plt.subplot(1, 2, 1)
            plt.hist(stats_df['alpha'], bins=25, histtype='step', color='black')
            plt.grid()
            plt.xlabel("$\\alpha$")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.hist(stats_df['r'], bins=100, histtype='step', color='black')
            plt.grid()
            plt.xlabel("$r$")
            plt.ylabel("Count")

            plt.savefig(output_path.joinpath("alpha_distribution.png"), dpi=150.)
            plt.close()

            # # Residuals distribution
            # plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))

            # plt.subplot(1, 2, 1)
            # plt.hist(wcs_residuals_df['x'], bins=200, histtype='step', color='black')
            # plt.grid()
            # plt.xlabel("Residuals $x$ [pixel]")
            # plt.ylabel("Count")

            # plt.subplot(1, 2, 2)
            # plt.hist(wcs_residuals_df['y'], bins=200, histtype='step', color='black')
            # plt.grid()
            # plt.xlabel("Residuals $y$ [pixel]")
            # plt.ylabel("Count")

            # plt.savefig(output_path.joinpath("wcs_residuals.png"), dpi=200.)
            # plt.close()
