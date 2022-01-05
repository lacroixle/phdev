#!/usr/bin/env python3

import sys
import pathlib
import shutil

import pandas as pd
import ztfimg.science
import ztfquery.io

sn_list_file = pathlib.Path(sys.argv[1])
lc_dir = pathlib.Path(sys.argv[2])
poloka_dir = pathlib.Path(sys.argv[3])

sn_df = pd.read_csv(sn_list_file, sep=",", comment="#")
sn_df.set_index("ztfname", inplace=True)

copy_files = True

zfilters = ['zg', 'zr', 'zi']

for sn in sn_df.index:
    poloka_dir.joinpath("{}".format(sn)).mkdir(exist_ok=True)

    def _create_subfolders(zfilter):
        lc_df = pd.read_hdf(lc_dir.joinpath("{}.hd5".format(sn)), key='lc_{}'.format(zfilter))

        for sciimg_filename_fits in lc_df['ipac_file']:
            # First create filter path

            poloka_dir.joinpath("{}/{}".format(sn, zfilter)).mkdir(exist_ok=True)
            folder_name = sciimg_filename_fits[:37]
            folder_path = poloka_dir.joinpath("{}/{}/{}".format(sn, zfilter, folder_name))

            folder_path.mkdir(exist_ok=True)
            folder_path.joinpath(".dbstuff").touch()

            sciimg_path = pathlib.Path(ztfquery.io.get_file(sciimg_filename_fits, downloadit=False, suffix='sciimg.fits'))
            mskimg_path = patylib.Path(ztfquery.io.get_file(sciimg_filename_fits, downloadit=False, suffix='mskimg.fits'))

            if copy_files:
                shutil.copy2(sciimg_path, folder_path)
                shutil.copy2(mskimg_path, folder_path)


    with pd.HDFStore(lc_dir.joinpath("{}.hd5".format(sn)), mode='r') as hdfstore:
        for zfilter in zfilters:
            if '/lc_{}'.format(zfilter) in hdfstore.keys():
                _create_subfolders(zfilter)
