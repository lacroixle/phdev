#!/usr/bin/env python3

import argparse
import pathlib

import pandas as pd
import numpy as np
from ztfimg.science import ScienceQuadrant
from ztfimg.stamps import stamp_it
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from astropy.io import fits

from utils import filtercodes


def sample_quadrant_tanspace(quadrant, wcs, center, stamp_size):
    center_px = center.to_pixel(wcs)
    pixels = np.stack([np.meshgrid(np.linspace(center_px[0] - stamp_size/2., center_px[0] + stamp_size/2., stamp_size),
                                   np.linspace(center_px[1] - stamp_size/2., center_px[1] + stamp_size/2., stamp_size))], axis=0).reshape(2, stamp_size, stamp_size).T.reshape(stamp_size**2, 2)

    return wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])


def sample_quadrant(quadrant, wcs, tanspace_radec, stamp_size, gaussian_blur_sigma=0.):
    pixels = tanspace_radec.to_pixel(wcs)

    # if gaussian_blur_sigma > 1e-3:
    #     quadrant = gaussian_filter(quadrant, gaussian_blur_sigma)

    interpolator = RegularGridInterpolator([np.arange(0., wcs.array_shape[1]), np.arange(0., wcs.array_shape[0])], quadrant.T, method='linear', bounds_error=False, fill_value=0.)

    return interpolator(pixels).reshape(stamp_size, stamp_size).T


def get_t0_quadrant(hdfstore, filtercode):
    sn_parameters = hdfstore.get('/params_{}'.format(filtercode))
    t0 = sn_parameters['t_0'].item()

    lc_quadrants = hdfstore.get('/lc_{}'.format(filtercode))
    t0_idx = np.argmin(np.abs(lc_quadrants.index - t0))

    t0_quadrant_name = lc_quadrants.iloc[t0_idx]['ipac_file']

    z = ScienceQuadrant.from_filename(t0_quadrant_name + "_sciimg.fits")
    t0_quadrant = np.asarray(z.get_data(applymask=False))
    wcs = z.wcs

    return t0_quadrant, wcs, z.header['seeing']


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--lc-folder', type=pathlib.Path, required=False)
    argparser.add_argument('--ztfname', type=pathlib.Path, required=False)
    argparser.add_argument('--stamp-size', type=int, default=32)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('-j', type=int, default=1)

    args = argparser.parse_args()
    args.lc_folder = args.lc_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if args.lc_folder and not args.ztfname:
        lc_files = args.lc_folder.glob("*.hd5")
    else:
        if args.ztfname.stem == str(args.ztfname):
            ztfnames = [args.ztfname]
        else:
            args.ztfname = args.ztfname.expanduser().resolve()
            with open(args.ztfname, 'r') as f:
                ztfnames = [ztfname.strip() for ztfname in f.readlines() if not ztfname.strip()[0] == "#"]

        lc_files = ["{}.hd5".format(ztfname) for ztfname in ztfnames]

    def _extract_colored_t0_stamp(lc_file):
        stamps = {}

        with pd.HDFStore(args.lc_folder.joinpath(lc_file)) as hdfstore:
            sn_info = hdfstore.get('/sn_info')
            sn_skycoord = SkyCoord(sn_info['sn_ra'].item(), sn_info['sn_dec'].item(), unit='deg')

            # Get first band with data
            tanspace_radec = None
            for filtercode in filtercodes:
                if '/lc_{}'.format(filtercode) in hdfstore.keys():
                    t0_quadrant, wcs, _ = get_t0_quadrant(hdfstore, filtercode)
                    tanspace_radec = sample_quadrant_tanspace(t0_quadrant, wcs, sn_skycoord, args.stamp_size)
                    break

            if not tanspace_radec:
                return

            for filtercode in filtercodes:
                if '/lc_{}'.format(filtercode) in hdfstore.keys():
                    t0_quadrant, wcs, _ = get_t0_quadrant(hdfstore, filtercode)
                    stamps[filtercode] = sample_quadrant(t0_quadrant, wcs, tanspace_radec, args.stamp_size)
                else:
                    stamps[filtercode] = None

        for filtercode in filtercodes:
            if stamps[filtercode] is None:
                stamps[filtercode] = np.zeros([args.stamp_size, args.stamp_size])

        # color_stamp = make_lupton_rgb(stamps['zr'], stamps['zg'], np.zeros([args.stamp_size, args.stamp_size]), stretch=200., Q=10.)
        color_stamp = make_lupton_rgb(stamps['zi'], stamps['zr'], stamps['zg'], stretch=100., Q=10.)
        plt.figure(figsize=(10., 10.), tight_layout=True)
        plt.imshow(color_stamp)
        plt.savefig(args.output.joinpath("{}.png".format(sn_info['ztfname'].item())), dpi=200.)
        plt.close()

        print(".", end="", flush=True)


    Parallel(n_jobs=args.j)(delayed(_extract_colored_t0_stamp)(lc_file) for lc_file in lc_files)
    print("")
