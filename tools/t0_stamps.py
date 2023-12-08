#!/usr/bin/env python3

import argparse
import pathlib
import math
import traceback

import pandas as pd
import numpy as np
from ztfimg.science import ScienceQuadrant
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS

from utils import filtercodes


def sample_quadrant_tanspace(wcs, center, stamp_size):
    center_px = center.to_pixel(wcs)
    # pixels = np.stack([np.meshgrid(np.linspace(center_px[0] - stamp_size/2., center_px[0] + stamp_size/2., stamp_size),
    #                                np.linspace(center_px[1] - stamp_size/2., center_px[1] + stamp_size/2., stamp_size))], axis=0).reshape(2, stamp_size, stamp_size).T.reshape(stamp_size**2, 2)

    # pixels = np.stack([np.meshgrid(np.linspace(center_px[0] - stamp_size/2., center_px[0] + stamp_size/2., stamp_size),
    #                                np.linspace(center_px[1] - stamp_size/2., center_px[1] + stamp_size/2., stamp_size), indexing='xy')], axis=0).reshape(2, stamp_size, stamp_size).T.reshape(stamp_size**2, 2)
    pixels = np.stack([np.meshgrid(np.linspace(center_px[0] - stamp_size/2., center_px[0] + stamp_size/2., stamp_size),
                                   np.linspace(center_px[1] - stamp_size/2., center_px[1] + stamp_size/2., stamp_size), indexing='xy')], axis=0).T.reshape(stamp_size**2, 2)
    return wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])


def sample_quadrant(quadrant, wcs, zp, tanspace_radec, stamp_size, sn, gaussian_blur_sigma=0.):
    pixels = np.array(tanspace_radec.to_pixel(wcs, origin=0, mode='all')).T
    # pixels[:, [0, 1]] = pixels[:, [1, 0]]
    sn_pixel = sn.to_pixel(wcs)

    interpolator = RegularGridInterpolator([np.arange(0., wcs.array_shape[1]), np.arange(0., wcs.array_shape[0])], quadrant.T, method='linear', bounds_error=False, fill_value=0.)

    return interpolator(pixels).reshape(stamp_size, stamp_size)
    # return 10**(0.4*(-2.5*np.log10(interpolator(pixels).reshape(stamp_size, stamp_size)) - zp))


def get_t0_quadrant(hdfstore, filtercode):
    sn_parameters = hdfstore.get('/params_{}'.format(filtercode))
    t0 = sn_parameters['t_0'].item()

    lc_quadrants = hdfstore.get('/lc_{}'.format(filtercode))
    t0_idx = np.argmin(np.abs(lc_quadrants.index - t0))

    t0_quadrant_name = lc_quadrants.iloc[t0_idx]['ipac_file']

    z = ScienceQuadrant.from_filename(t0_quadrant_name + "_sciimg.fits")
    with fits.open(z._filepath) as hdul:
        wcs = WCS(hdul[0].header)
    t0_quadrant = np.asarray(z.get_data(applymask=True))
    # wcs = z.wcs

    return t0_quadrant, wcs, float(z.header['seeing']), float(z.header['magzp'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--lc-folder', type=pathlib.Path, required=False)
    argparser.add_argument('--ztfname', type=pathlib.Path, required=False)
    argparser.add_argument('--stamp-size', type=int, default=32)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('--target', choices=['host', 'sn'], default='sn')
    argparser.add_argument('--arrow', action='store_true')
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

        try:
            with pd.HDFStore(args.lc_folder.joinpath(lc_file)) as hdfstore:
                sn_info = hdfstore.get('/sn_info')
                target = 'sn'
                if args.target == 'host':
                    target = 'host'
                object_skycoord = SkyCoord(sn_info['{}_ra'.format(target)].item(), sn_info['{}_dec'.format(target)].item(), unit='deg')

                # Get first band with data
                tanspace_radec = None
                for filtercode in filtercodes:
                    if '/lc_{}'.format(filtercode) in hdfstore.keys():
                        _, wcs, _, _ = get_t0_quadrant(hdfstore, filtercode)
                        tanspace_radec = sample_quadrant_tanspace(wcs, object_skycoord, args.stamp_size)
                        break

                if not tanspace_radec:
                    return

                if args.arrow:
                    arrow_length = 20.
                    arrow_dl = 4.
                    arrow_dir = np.array([-1/math.sqrt(2)*arrow_length]*2)
                    arrow_tail = np.array([args.stamp_size/2. + 1/math.sqrt(2)*(arrow_length + arrow_dl)]*2)
                    if args.target == 'host':
                        sn_skycoord = SkyCoord(sn_info['sn_ra'].item(), sn_info['sn_dec'].item(), unit='deg')
                        host_skycoord = SkyCoord(sn_info['host_ra'].item(), sn_info['host_dec'].item(), unit='deg')
                        sn_px = np.stack(sn_skycoord.to_pixel(wcs))
                        host_px = np.stack(host_skycoord.to_pixel(wcs))
                        d = sn_px - host_px
                        arrow_tail += d

                for filtercode in filtercodes:
                    #if '/lc_{}'.format(filtercode) in hdfstore.keys():
                    if filtercode == 'zg':
                        t0_quadrant, wcs, _, zp = get_t0_quadrant(hdfstore, filtercode)
                        print(filtercode)
                        stamps[filtercode] = sample_quadrant(t0_quadrant, wcs, zp, tanspace_radec, args.stamp_size, object_skycoord)
                        # plt.imshow(stamps[filtercode])
                        # plt.show()
                    else:
                        stamps[filtercode] = None

            for filtercode in filtercodes:
                if stamps[filtercode] is None:
                    stamps[filtercode] = np.zeros([args.stamp_size, args.stamp_size])

            # color_stamp = make_lupton_rgb(stamps['zr'], stamps['zg'], np.zeros([args.stamp_size, args.stamp_size]), stretch=10., Q=1.)
            color_stamp = make_lupton_rgb(stamps['zi'], stamps['zr'], stamps['zg'], stretch=100., Q=10.)
            #plt.imshow(stamps['zg'])
            plt.imshow(np.log10(stamps['zg']))
            plt.show()
            return
            plt.figure(figsize=(10., 10.), tight_layout=True)
            fig = plt.imshow(color_stamp)
            if args.arrow:
                plt.arrow(arrow_tail[0], arrow_tail[1], arrow_dir[0], arrow_dir[1], head_width=3., length_includes_head=True, color='white')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.axis('off')
            #plt.savefig(args.output.joinpath("{}_{}.png".format(sn_info['ztfname'].item(), args.stamp_size)), dpi=150., pad_inches=0., bbox_inches='tight')
            plt.savefig(args.output.joinpath("{}.png".format(sn_info['ztfname'].item())), dpi=400., pad_inches=0., bbox_inches='tight')
            plt.show()
            plt.close()

            print(".", end="", flush=True)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            traceback.print_exc()
            print("x", end="", flush=True)


    Parallel(n_jobs=args.j)(delayed(_extract_colored_t0_stamp)(lc_file) for lc_file in lc_files)
    print("")
