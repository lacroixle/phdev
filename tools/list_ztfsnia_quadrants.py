#!/usr/bin/env python3

import pathlib
import argparse

import pandas as pd

import utils


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--lc-folder', type=pathlib.Path)
    argparser.add_argument('--ztfname', type=str)

    args = argparser.parse_args()
    args.lc_folder = args.lc_folder.expanduser().resolve()

    if args.ztfname:
        lc_files = [args.lc_folder.joinpath("{}.hd5".format(args.ztfname))]
    else:
        lc_files = [args.lc_folder.glob("*.hd5")]

    quadrant_list = []
    for lc_file in lc_files:
        with pd.HDFStore(lc_file, mode='r') as hdfstore:
            for filtercode in utils.filtercodes:
                if '/lc_{}'.format(filtercode) in hdfstore.keys():
                    lc_df = pd.read_hdf(hdfstore, key='lc_{}'.format(filtercode))
                    quadrant_list.extend(lc_df['ipac_file'])

    for quadrant in quadrant_list:
        print(quadrant)
