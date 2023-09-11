#!/usr/bin/env python3

import pathlib
import argparse

from yaml import Loader, load
import pandas as pd
import sncosmo
import matplotlib.pyplot as plt


from utils import filtercodes, ListTable
from lightcurve import Lightcurve


def retrieve_redshift(ztfname, lc_folder):
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')
    return sn_parameters['redshift']


def plot_lightcurve(ztfname):
    sncosmo_dfs = []
    for filtercode in filtercodes:
        lightcurve = Lightcurve(ztfname, filtercode, args.wd)
        if not lightcurve.path.exists():
            continue

        if not lightcurve.func_status('smphot') or not lightcurve.func_status('calib'):
            continue

        df = ListTable.from_filename(lightcurve.smphot_path.joinpath("lightcurve_sn.dat")).df
        with open(lightcurve.path.joinpath("lightcurve.yaml"), 'r') as f:
            lightcurve_globs = load(f, Loader=Loader)
            zp = lightcurve_globs['calib']['zp']

        sncosmo_df = pd.DataFrame({'time': df['mjd'], 'flux': df['flux'], 'fluxerr': df['varflux']})
        sncosmo_df['zp'] = -zp
        sncosmo_df['band'] = 'ztf' + filtercode[1]
        sncosmo_df['zpsys'] = 'ab'

        sncosmo_dfs.append(sncosmo_df)


    z = retrieve_redshift(lightcurve.name, args.lc_folder)

    sncosmo_df = pd.concat(sncosmo_dfs)

    model = sncosmo.Model(source='salt2')
    model.set(z=z)
    result, fitted_model = sncosmo.fit_lc(sncosmo_df.to_records(), model, ['t0', 'x0', 'x1', 'c'])

    sncosmo.plot_lc(sncosmo_df.to_records(), fitted_model, errors=result.errors)
    plt.savefig(args.output.joinpath("{}.png".format(ztfname)), dpi=300.)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, required=True)
    argparser.add_argument('--wd', type=pathlib.Path, required=True)
    argparser.add_argument('--lc-folder', type=pathlib.Path, required=True)
    argparser.add_argument('--output', type=pathlib.Path, required=True)
    argparser.add_argument('-j', type=int, default=1)

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()
    args.lc_folder = args.lc_folder.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    ztfnames = None
    if args.ztfname is not None:
        if args.ztfname.stem == str(args.ztfname):
            ztfnames = [str(args.ztfname)]
        else:
            args.ztfname = args.ztfname.expanduser().resolve()
            if args.ztfname.exists():
                with open(args.ztfname, 'r') as f:
                    ztfnames = [ztfname[:-1] for ztfname in f.readlines()]
            else:
                pass

    print("Found {} SN1a".format(len(ztfnames)))

    for ztfname in ztfnames:
        plot_lightcurve(ztfname)
