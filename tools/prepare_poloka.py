#!/usr/bin/env python3

import sys
import pathlib
import shutil

import pandas as pd
import ztfimg.science
import ztfquery.io
import joblib
import numpy as np
from astropy.io import fits

sn_list_file = pathlib.Path(sys.argv[1])
lc_dir = pathlib.Path(sys.argv[2])
poloka_dir = pathlib.Path(sys.argv[3])

sn_df = pd.read_csv(sn_list_file, sep=",", comment="#")
sn_df.set_index("ztfname", inplace=True)

print("Found {} SNe 1a...".format(len(sn_df)), flush=True)

copy_files = True

zfilters = ['zg', 'zr', 'zi']

for sn in sn_df.index:
    print("In SN {}".format(sn))
    poloka_dir.joinpath("{}".format(sn)).mkdir(exist_ok=True)

    def _create_subfolders(zfilter, hdfstore):
        print("In filter subfolder {}".format(zfilter))
        poloka_dir.joinpath("{}/{}".format(sn, zfilter)).mkdir(exist_ok=True)
        lc_df = pd.read_hdf(hdfstore, key='lc_{}'.format(zfilter))

        def _create_subfolder(sciimg_filename):
            # Check files exist
            sciimg_path = pathlib.Path(ztfquery.io.get_file(sciimg_filename, downloadit=False, suffix='sciimg.fits'))
            mskimg_path = pathlib.Path(ztfquery.io.get_file(sciimg_filename, downloadit=False, suffix='mskimg.fits'))
            if not sciimg_path.exists() or not mskimg_path.exists():
                print("Fail: {}".format(sciimg_filename))
                return

            # First create filter path
            poloka_dir.joinpath("{}/{}".format(sn, zfilter)).mkdir(exist_ok=True)
            folder_name = sciimg_filename[:37]
            folder_path = poloka_dir.joinpath("{}/{}/{}".format(sn, zfilter, folder_name))

            folder_path.mkdir(exist_ok=True)
            folder_path.joinpath(".dbstuff").touch()

            def _create_symlink(path, symlink_to):
                if path.exists():
                    path.unlink()

                path.symlink_to(symlink_to)

            _create_symlink(folder_path.joinpath("elixir.fits"), sciimg_path)
            #_create_symlink(folder_path.joinpath("mask.fits"), mskimg_path)

            # Dead mask
            z = ztfimg.science.ScienceQuadrant.from_filename(sciimg_path)

            deads = np.array(z.get_mask(tracks=False, ghosts=False, spillage=False, spikes=False, dead=True,
                                        nan=False, saturated=False, brightstarhalo=False, lowresponsivity=False,
                                        highresponsivity=False, noisy=False, verbose=False), dtype=np.uint8)

            mskhdu = fits.PrimaryHDU([deads])
            mskhdu.writeto(folder_path.joinpath("deads.fits.gz"), overwrite=True)

            print("Success: {}, dead pixel count={}".format(sciimg_filename, np.sum(deads)))
            #exit()

        #for sciimg_filename_fits in lc_df['ipac_file']:
        joblib.Parallel(n_jobs=int(sys.argv[4]))(joblib.delayed(_create_subfolder)(sciimg_filename) for sciimg_filename in lc_df['ipac_file'])


    if lc_dir.joinpath("{}.hd5".format(sn)).exists():
        with pd.HDFStore(lc_dir.joinpath("{}.hd5".format(sn)), mode='r') as hdfstore:
            for zfilter in zfilters:
                if '/lc_{}'.format(zfilter) in hdfstore.keys():
                    _create_subfolders(zfilter, hdfstore)
