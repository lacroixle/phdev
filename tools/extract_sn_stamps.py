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
argparser.add_argument('--create-folder', dest='create_folder', action='store_true')
argparser.add_argument('--stamp-size', dest='stamp_size', type=int, default=20)

args = argparser.parse_args()

zfilter = args.zfilter
lc_folder = args.lc_folder.expanduser().resolve()
n_jobs = args.n_jobs
output_folder = args.output.expanduser().resolve()
ztfname = args.ztfname
create_folder = args.create_folder
stamp_size = args.stamp_size

if args.cosmodr:
    cosmo_dr_folder = args.cosmodr
elif 'COSMO_DR_FOLDER' in os.environ.keys():
    cosmo_dr_folder = pathlib.Path(os.environ.get("COSMO_DR_FOLDER"))
else:
    print("Cosmo DR folder not set! Either set COSMO_DR_FOLDER environnement variable or use the --cosmodr parameter.")
    exit(-1)


print("Extracting SN sequence for {} with filter {}".format(ztfname, zfilter))
with pd.HDFStore(lc_folder.joinpath("{}.hd5".format(ztfname)), mode='r') as hdfstore:
   if '/lc_{}'.format(zfilter) in hdfstore.keys():
      lc_df = pd.read_hdf(hdfstore, key='lc_{}'.format(zfilter))
      print("Found {} quadrants".format(len(lc_df)))
   else:
      print("No data for filter {}".format(zfilter))
      exit()


coords_file = cosmo_dr_folder.joinpath("ztfdr2_coords.csv")
coords_df = pd.read_csv(coords_file, sep=" ", index_col='ztfname')

sn_ra = coords_df.loc[ztfname]['host_ra']
sn_dec = coords_df.loc[ztfname]['host_dec']

def extract_stamp(sciimg_file):
   quadrant = ztfimg.science.ScienceQuadrant.from_filename(sciimg_file)

   x, y = quadrant.wcs.world_to_pixel(astropy.coordinates.SkyCoord(sn_ra, sn_dec, unit='deg'))
   print(".", end="", flush=True)

   return np.asarray(ztfimg.stamps.stamp_it(quadrant.get_data('clean'), x, y, stamp_size, asarray=True))

print("Extracting SN stamp sequence")
stamps = Parallel(n_jobs=n_jobs)(delayed(extract_stamp)(sciimg_file) for sciimg_file in lc_df)

print("")
print("Building stamp tensor")
stamps = np.nan_to_num(np.stack(stamps))

print("Normalizing")
stamps = (stamps - np.min(stamps))/(np.max(stamps) - np.min(stamps))
stamps = (stamps*255.).astype(np.uint8)

print("Saving")
if create_folder:
   output_folder.joinpath(ztfname).mkdir(exist_ok=True)
   output_folder = output_folder.joinpath("{}/{}".format(ztfname, zfilter))
   output_folder.mkdir(exist_ok=True)

for i, stamp in enumerate(stamps):
   img = Image.fromarray(stamp)
   img.save(output_folder.joinpath("{}.png".format(i)))

print("Done")
