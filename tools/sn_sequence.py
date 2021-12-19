#!/usr/bin/env python3

import sys
import pathlib
import argparse
import os

import ztfimg.science
import ztfimg.stamps
import pandas as pd
from joblib import Parallel, delayed
import astropy.coordinates
import numpy as np
from PIL import Image


zfilters = ['zr', 'zg', 'zi']


argparser = argparse.ArgumentParser(description="SN stamp sequence extractor.")
argparser.add_argument("--output", type=pathlib.Path, help="Output folder.", required=True)
argparser.add_argument("--ztfname", type=str, help="SN to process.", required=True)
argparser.add_argument('--zfilter', type=str, choices=zfilters, required=True)
argparser.add_argument('-j', dest='n_jobs', type=int, default=1)
argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path, required=True)
argparser.add_argument('--cosmodr', type=pathlib.Path)

args = argparser.parse_args()

ztfname = args.ztfname
zfilter = args.zfilter
lc_folder = args.lc_folder.expanduser().resolve()
n_jobs = args.n_jobs

if args.cosmodr:
    cosmo_dr_folder = args.cosmodr
elif 'COSMO_DR_FOLDER' in os.environ.keys():
    cosmo_dr_folder = pathlib.Path(os.environ.get("COSMO_DR_FOLDER"))
else:
    print("Cosmo DR folder not set! Either set COSMO_DR_FOLDER environnement variable or use the --cosmodr parameter.")
    exit(-1)

coords_file = cosmo_dr_folder.joinpath("ztfdr2_coords.csv")


lc_df = pd.read_hdf(lc_folder.joinpath("{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter))
coords_df = pd.read_csv(coords_file, sep=" ", index_col='ztfname')

sn_ra = coords_df.loc[ztfname]['host_ra']
sn_dec = coords_df.loc[ztfname]['host_dec']

def extract_stamp(sciimg_file):
   quadrant = ztfimg.science.ScienceQuadrant.from_filename(sciimg_file)

   x, y = quadrant.wcs.world_to_pixel(astropy.coordinates.SkyCoord(sn_ra, sn_dec, unit='deg'))
   print(".", end="", flush=True)

   return np.asarray(ztfimg.stamps.stamp_it(quadrant.get_data('clean'), x, y, 40, asarray=True))

stamps = Parallel(n_jobs=n_jobs)(delayed(extract_stamp)(sciimg_file) for sciimg_file in lc_df)

print("")
print("Building stamp tensor...")
stamps = np.nan_to_num(np.stack(stamps))
print(np.min(stamps), np.max(stamps))

print("Normalizing...")
#stamps = (stamps*np.nanmax(stamps)).astype(np.uint8)
stamps = (stamps - np.min(stamps))/(np.max(stamps) - np.min(stamps))
stamps = (stamps*255.).astype(np.uint8)
print(np.min(stamps), np.max(stamps))
#print(stamps)

print("Saving")
for i, stamp in enumerate(stamps):
   img = Image.fromarray(stamp)
   img.save("sn/{}.png".format(i))
   print(".", end="", flush=True)

print()


