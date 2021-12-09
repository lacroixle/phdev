#!/usr/bin/env python3

import sys
import pathlib

import ztfimg.science
import pandas as pd
from joblib import Parallel, delayed

ztfname = sys.argv[1]
zfilter = sys.argv[2]
data_folder = pathlib.Path(sys.argv[3])

lc_df = pd.read_hdf(data_folder.joinpath("ztf/lc/{}.hd5".format(ztfname)), key='lc_{}'.format(zfilter))
coords_df = pd.read_csv(data_folder.joinpath("ztf/ztfcosmoidr/dr2/ztfdr2_coords.csv"), sep=" ", index_col='ztfname')

sn_ra = coords_df.loc[ztfname]['sn_ra']
sn_dec = coords_df.loc[ztfname]['sn_dec']

def extract_stamp(sciimg_file):
   quadrant = ztfimg.science.ScienceQuadrant.from_filename(sciimg_file)

   x, y = quadrant.wcs.world_to_pixel(sn_ra, sn_dec)

   print(x, y)


Parallel(n_jobs=1)(delayed(extract_stamp)(sciimg_file) for sciimg_file in lc_df)
