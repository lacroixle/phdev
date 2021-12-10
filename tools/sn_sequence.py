#!/usr/bin/env python3

import sys
import pathlib

import ztfimg.science
import ztfimg.stamps
import pandas as pd
from joblib import Parallel, delayed
import astropy.coordinates
import numpy as np
from PIL import Image

ztfname = sys.argv[1]
zfilter = sys.argv[2]
lc_folder = pathlib.Path(sys.argv[3])
coords_file = pathlib.Path(sys.argv[4])

lc_df = pd.read_hdf(lc_folder.joinpath("{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter))
coords_df = pd.read_csv(coords_file, sep=" ", index_col='ztfname')

sn_ra = coords_df.loc[ztfname]['sn_ra']
sn_dec = coords_df.loc[ztfname]['sn_dec']

def extract_stamp(sciimg_file):
   quadrant = ztfimg.science.ScienceQuadrant.from_filename(sciimg_file)

   x, y = quadrant.wcs.world_to_pixel(astropy.coordinates.SkyCoord(sn_ra, sn_dec, unit='deg'))
   print(".", end="", flush=True)

   return ztfimg.stamps.stamp_it(quadrant.get_data('clean'), x, y, 20).data

stamps = Parallel(n_jobs=8)(delayed(extract_stamp)(sciimg_file) for sciimg_file in lc_df.iloc[:10])

print("")
print("Building stamp tensor...")
stamps = np.stack([np.array(stamp) for stamp in stamps])

print(stamps)

print("Normalizing...")
stamps = (stamps*np.nanmax(stamps)).astype(np.uint8)

print(stamps)

print("Saving")
for i, stamp in enumerate(stamps):
   img = Image.fromarray(stamp)
   img.save("sn/{}.png".format(i))
   print(".", end="", flush=True)

print()


