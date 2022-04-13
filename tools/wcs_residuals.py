#!/usr/bin/env python3

import argparse
import pathlib
import os
import time
import itertools

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import pyloka
from scipy.stats import linregress

import list_format
import utils


filtercodes = ['zg', 'zr', 'zi']

def residuals_quadrant(quadrant_path, reference_quadrant_path, gaia_stars, output_path):
    # Astrometric residuals of psfstars with Gaia stars
    quadrant_name = quadrant_path.name
    print(quadrant_name)

    if not args.wd.joinpath("{}/{}/pmfit/transfoTo{}.dat".format(ztfname, filtercode, quadrant_name)).exists():
        print("Could not find transformation!")
        return

    with open(quadrant_path.joinpath("psfstars.list"), 'r') as f:
        _, stars = utils.read_list(f)

    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        wcs = WCS(hdul[0].header)
        seeing = hdul[0].header['seeing']

    cosmic_count = utils.ListTable.from_filename(quadrant_path.joinpath("se.list")).header['cosmic_count']

    aperse_stars_count = utils.get_cat_size(quadrant_path.joinpath("aperse.list"))
    standalone_stars_count = utils.get_cat_size(quadrant_path.joinpath("standalone_stars.list"))
    psf_stars_count = utils.get_cat_size(quadrant_path.joinpath("psfstars.list"))

    gaia_coords_radec = SkyCoord(gaia_stars['ra'], gaia_stars['dec'], unit='deg')
    gaia_coords_radec = utils.contained_in_exposure(gaia_coords_radec, wcs)
    x, y = gaia_coords_radec.to_pixel(wcs)
    # x, y = pyloka.radec2pix(str(quadrant_path.joinpath("calibrated.fits")), gaia_coords_radec.frame.data.lon.value, gaia_coords_radec.frame.data.lat.value)
    gaia_coords_pixel = pd.DataFrame({'x': x, 'y': y}).to_records(index=False)
    #gaia_coords_pixel = pd.DataFrame({'x': gaia_coords_pixel[0], 'y': gaia_coords_pixel[1]}).dropna().to_records(index=False)
    # utils.cat_to_ds9regions(gaia_coords_pixel, "out.reg")
    # utils.cat_to_ds9regions(stars.to_records(), "out2.reg", color='red')

    # Translate the gaia stars catalogue into quadrant pixel space
    i = utils.match_pixel_space(gaia_coords_pixel, stars, radius=3.)

    refstars = gaia_coords_pixel[i[i>=0]]
    refstars_pair = stars[i>=0]
    gaia_stars_count = len(refstars)

    plt.subplots(1, 2, figsize=(10., 5.))
    plt.suptitle("WCS with Gaia stars residuals")
    plt.subplot(1, 2, 1)
    plt.hist(refstars['x'] - refstars_pair['x'], bins=100, histtype='step', color='black')
    plt.grid()
    plt.xlabel("Residual [pixel]")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(refstars['y'] - refstars_pair['y'], bins=100, histtype='step', color='black')
    plt.xlabel("Residual [pixel]")
    plt.grid()

    plt.savefig(output_path.joinpath("{}_wcs_gaia_residuals.png".format(quadrant_name)), dpi=150.)
    plt.close()

    # Check photometric linearity
    with open(reference_quadrant_path.joinpath("psfstars.list"), 'r') as f:
        _, reference_stars = utils.read_list(f)

    try:
        pol = utils.poly2d_from_file(args.wd.joinpath("{}/{}/pmfit/transfoTo{}.dat".format(ztfname, filtercode, quadrant_name)))
    except:
        print("Could not find transformation!")
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
        print(match_stars_count)
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

    return alpha, rvalue, gaia_stars_count, match_stars_count, aperse_stars_count, standalone_stars_count, psf_stars_count, seeing, cosmic_count

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)

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

    print("Found {} SN1a".format(len(ztfnames)))


    for ztfname in ztfnames:
        print("For {}".format(ztfname))
        for filtercode in filtercodes:
            print("In filter band {}".format(filtercode))
            band_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not band_folder.exists():
                continue

            # Get reference quadrant
            reference_quadrant_path = utils.get_ref_quadrant_from_driver(args.wd.joinpath("{}/{}/{}_driver_{}".format(ztfname, filtercode, ztfname, filtercode)))
            if not reference_quadrant_path:
                print("{}-{}: no reference quadrant found!".format(ztfname, filtercode))
                continue

            print("Reference exposure={}".format(reference_quadrant_path.name))
            gaia_stars = np.load(band_folder.joinpath("gaia.npy"))

            quadrant_paths = [quadrant_path for quadrant_path in list(band_folder.glob("ztf_*")) if quadrant_path.is_dir() and quadrant_path.joinpath("psfstars.list").exists()]

            output_path = args.output.joinpath("{}/{}".format(ztfname, filtercode))
            os.makedirs(output_path, exist_ok=True)

            alphas = [residuals_quadrant(quadrant_path, reference_quadrant_path, gaia_stars, output_path) for quadrant_path in quadrant_paths]
            alphas_mask = [alpha is not None for alpha in alphas]
            alphas = list(itertools.compress(alphas, alphas_mask))
            quadrant_paths = list(itertools.compress(quadrant_paths, alphas_mask))
            alpha_df = pd.DataFrame(data=np.array(alphas), index=list(map(lambda x: x.name, quadrant_paths)), columns=['alpha', 'r', 'n_gaia', 'n_match', 'n_aperse', 'n_standole', 'n_psf', 'seeing', 'n_cosmic'])
            alpha_df.index.name = 'quadrant'

            alpha_df.to_csv(band_folder.joinpath("alpha.csv"), sep=",")

            plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))

            plt.subplot(1, 2, 1)
            plt.hist(alpha_df['alpha'], bins=25, histtype='step', color='black')
            plt.grid()
            plt.xlabel("$\\alpha$")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.hist(alpha_df['r'], bins=100, histtype='step', color='black')
            plt.grid()
            plt.xlabel("$r$")
            plt.ylabel("Count")

            plt.savefig(output_path.joinpath("alpha_distribution.png"), dpi=150.)
            plt.close()
            plt.show()
