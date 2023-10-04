#!/usr/bin/env python3

import argparse
import pathlib

from ztfquery.io import get_file
import ztfimg
import pandas as pd
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from utils import quadrant_name_explode

def rawquadrant_to_overscan_df(rawimg_filename, qids):
    if args.absolute_path:
        rawimg_path = pathlib.Path(rawimg_filename)
    else:
        rawimg_path = pathlib.Path(get_file(rawimg_filename, downloadit=False))

    if not rawimg_path.exists():
        #raise FileNotFoundError("Could not find raw img {}!".format(rawimg_path))
        print("Could not find raw img {}!".format(rawimg_path))
        return

    with fits.open(rawimg_path) as hdul:
        if len(hdul) != 9:
            print("Quadrant {} does not have overscan!".format(rawimg_path))
            return

        ccdid = int(hdul[0].header['CCD_ID'])
        temp = float(hdul[0].header['CCDTMP{}'.format(str(ccdid).zfill(2))])
        head_temp = float(hdul[0].header['HEADTEMP'])
        dewpressure = float(hdul[0].header['DEWPRESS'])
        detheat = float(hdul[0].header['DETHEAT'])
        airmass = float(hdul[0].header['AIRMASS'])
        moonillf = float(hdul[0].header['MOONILLF'])
        mjd = float(hdul[0].header['OBSMJD'])
        azimuth = float(hdul[0].header['AZIMUTH'])
        elevation = float(hdul[0].header['ELVATION'])


    rawimg = ztfimg.RawCCD.from_filename(str(rawimg_path), as_path=True)

    _, _, _, _, filterid, _ = quadrant_name_explode(rawimg_filename, kind='raw')

    dfs = []
    for qid in qids:
        quadrant = rawimg.get_quadrant(qid)
        data = quadrant.get_data(reorder=False)
        skylev = np.median(data)
        if qid in [1, 4]:
            overscan = quadrant.get_overscan(which='raw', userange=None)
            lastline = data[:, -1]
            mean_overscan = np.median(overscan, axis=0)
            N = len(mean_overscan)
            firstlines = data[:, 0:N]
            mean_trail_overscan = np.median(overscan[:, N-5:-1])
        else:
            overscan = quadrant.get_overscan(which='raw', userange=None)[:, ::-1]
            lastline = data[:, 0]
            mean_overscan = np.median(overscan, axis=0)
            N = len(mean_overscan)
            firstlines = data[:, -N:]
            firstlines = firstlines[:, ::-1]
            mean_trail_overscan = np.median(overscan[:, N-5:-1])

        mean_lastline = np.median(lastline)
        mean_firstlines = np.median(firstlines, axis=0)

        d = {'quadrant': [rawimg_filename]*N,
             'skylev': [skylev]*N,
             'ccdid': [ccdid]*N,
             'qid': [qid]*N,
             'fieldid': [int(rawimg_filename[19:25])]*N,
             'filterid': [filterid]*N,
             'temp': [temp]*N,
             'head_temp': [head_temp]*N,
             'dewpressure': [dewpressure]*N,
             'detheat': [detheat]*N,
             'airmass': [airmass]*N,
             'moonillf': [moonillf]*N,
             'mjd': [mjd]*N,
             'azimuth': [azimuth]*N,
             'elevation': [elevation]*N,
             'lastcol': [mean_lastline]*N,
             'j': np.arange(N),
             'overscan': mean_overscan,
             'firstlines': mean_firstlines,
             'mean_trail_overscan': [mean_trail_overscan]*N}

        dfs.append(pd.DataFrame.from_dict(d))

    print("{}-{} {} N={}".format(rawimg_filename, qids, rawimg_filename[19:25], N))

    return pd.concat(dfs)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ccd-list', type=pathlib.Path)
    argparser.add_argument('--output', type=pathlib.Path)
    argparser.add_argument('--qid', type=str)
    argparser.add_argument('--absolute-path', action='store_true')
    argparser.add_argument('-j', type=int, default=1)

    args = argparser.parse_args()
    args.ccd_list = args.ccd_list.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    with open(args.ccd_list, 'r') as f:
        ccd_list = list(map(lambda x: x.strip(), f.readlines()))

    if args.qid == 'all':
        qids = list(range(1, 5))
    else:
        qids = list(map(lambda x: int(x.strip()), args.qid.split(",")))

    dfs = Parallel(n_jobs=args.j)(delayed(rawquadrant_to_overscan_df)(ccd, qids) for ccd in ccd_list)
    dfs = list(filter(lambda x: isinstance(x, pd.DataFrame), dfs))
    dfs = pd.concat(dfs)
    dfs.to_parquet(args.output)
