#!/usr/bin/env python3

import argparse
import pathlib
import time
from itertools import chain

import numpy as np
from ztfimg import science
from ztfquery import io
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--quadrant-list', type=pathlib.Path)
    argparser.add_argument('-j', type=int)

    args = argparser.parse_args()
    args.quadrant_list = args.quadrant_list.expanduser().resolve()

    quadrant_list = []
    with open(args.quadrant_list, 'r') as f:
        quadrant_list.extend(map(lambda x: x.strip(), f.readlines()))

    print("{} quadrants to download".format(len(quadrant_list)))
    print("Dowloading sciimg")
    io.get_file(quadrant_list)

    print("Downloading mskimg")
    io.get_file(quadrant_list, suffix="mskimg.fits")

    print("Done")


    # # Filter out already downloaded quadrants

    # def _filter_downloaded_quadrants(quadrant_list):
    #     to_download = []
    #     for quadrant in quadrant_list:
    #         pass

    #     #return to_download
    #     return quadrant_list

    # print("Filtering out already downloaded quadrants")
    # start_time = time.perf_counter()

    # # Divide quadrant list into even chunks
    # quadrant_lists = [array.tolist() for array in np.array_split(quadrant_list, args.j)]

    # to_download_list = Parallel(n_jobs=args.j)(delayed(_filter_downloaded_quadrants)(quadrant_list) for quadrant_list in quadrant_lists)

    # to_download = list(chain(*to_download_list))
    # print(len(to_download))

    # print("Dond. Elapsed time={}".format(time.perf_counter()-start_time))
