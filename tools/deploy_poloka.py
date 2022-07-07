#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import datetime
import os
import time
import shutil
import sys
import socket
import copy
import traceback
import json

from joblib import Parallel, delayed
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, wait, get_worker
from dask_jobqueue import SLURMCluster, SGECluster
import ztfquery.io
import numpy as np
from skimage.morphology import label
from saunerie.plottools import binplot
from astropy.time import Time
import imageproc.composable_functions as compfuncs
import saunerie.fitparameters as fp
from scipy import sparse
from croaks import DataProxy


import utils

matplotlib.use('Agg')

ztf_filtercodes = ['zg', 'zr', 'zi', 'all']
poloka_func = []


def run_and_log(cmd, logger=None):
    if logger:
        logger.info("Running command: \"{}\"".format(" ".join([str(s) for s in cmd])))
        start_time = time.perf_counter()

    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)

    if logger:
        logger.info("Done running command. Elapsed time={}".format(time.perf_counter() - start_time))
        logger.info("Command stdout/stderr output:")
        logger.info(out.stdout)
        logger.info("=========================== output end ===========================")


    return out.returncode


def make_catalog(quadrant_folder, logger):
    logger.info("Retrieving calibrated.fits...")
    sciimg_path = ztfquery.io.get_file(quadrant_folder.name + "_sciimg.fits", downloadit=False)
    logger.info("Located at {}".format(sciimg_path))
    shutil.copyfile(sciimg_path, quadrant_folder.joinpath("calibrated.fits"))

    run_and_log(["make_catalog", quadrant_folder, "-O", "-S"], logger)

    return quadrant_folder.joinpath("se.list").exists()


poloka_func.append({'map': make_catalog})


def mkcat2(folder, logger):
    run_and_log(["mkcat2", folder, "-o"], logger)
    return folder.joinpath("standalone_stars.list").exists()


poloka_func.append({'map': mkcat2})


def makepsf(folder, logger):
    run_and_log(["makepsf", folder, "-f"], logger)
    return folder.joinpath("psfstars.list").exists()

poloka_func.append({'map': makepsf})


def pipeline(folder, logger):
    if not make_catalog(folder, logger):
        return False

    if not mkcat2(folder, logger):
        return False

    if not makepsf(folder, logger):
        return False

    return True

poloka_func.append({'map': pipeline})


#files_to_keep = ["elixir.fits", "mask.fits", "deads.fits.gz", ".dbstuff"]
files_to_keep = ["elixir.fits", "dead.fits.gz", ".dbstuff"]
def clean(folder, logger):
    # calibrated.fits header gets modified, it might be problematic at some point (or not)
    if args.dry_run:
        print("In quadrant folder {}".format(folder))

    files = list(folder.glob("*"))
    files_to_delete = [file_to_delete for file_to_delete in files if file_to_delete.name not in files_to_keep]

    for file_to_delete in files_to_delete:
        if not args.dry_run:
            file_to_delete.unlink()
        else:
            print("File to delete: {}".format(file_to_delete))

    return True


def clean_reduce(folder, ztfname, filtercode, logger):
    # Delete all files
    files_to_delete = list(filter(lambda f: f.is_file(), list(folder.glob("*"))))
    [f.unlink() for f in files_to_delete]

    shutil.rmtree(folder.joinpath("pmfit"))
    shutil.rmtree(folder.joinpath("pmfit_plot"))
    shutil.rmtree(folder.joinpath("smphot_output"))


poloka_func.append({'map': clean, 'reduce': clean_reduce})


# Extract data from standalone stars and plot several distributions
def stats(folder, logger):
    def _extract_from_list(list_filename, hdfstore):
        list_path = folder.joinpath(list_filename).with_suffix(".list")

        if not list_path.exists():
            return False

        with open(list_path, mode='r') as f:
            global_params, df, _, _ = utils.read_list(f)

        hdfstore.put(list_path.stem, df)
        hdfstore.put("{}_globals".format(list_path.stem), pd.DataFrame([global_params]))

        return True

    import warnings
    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

    with pd.HDFStore(folder.joinpath("lists.hdf5"), mode='w') as hdfstore:
        # From make_catalog
        _extract_from_list("se", hdfstore)

        # From mkcat2
        cont = _extract_from_list("standalone_stars", hdfstore)

        if not cont:
            return True

        _extract_from_list("aperse", hdfstore)

        # From calibrated.fits
        keywords = ['sexsky', 'sexsigma', 'bscale', 'bzero', 'origsatu', 'saturlev', 'backlev', 'back_sub', 'seseeing', 'gfseeing']

        calibrated = {}
        with fits.open(folder.joinpath("calibrated.fits")) as hdul:
            for keyword in keywords:
                calibrated[keyword] = hdul[0].header[keyword]

            hdfstore.put('calibrated', pd.DataFrame([calibrated]))

        # From makepsf
        cont = _extract_from_list("psfstars", hdfstore)

        if not cont:
            return True

        _extract_from_list("psftuples", hdfstore)

    return True


def stats_reduce(cwd, ztfname, filtercode, logger):
    # Seeing histogram
    folders = [folder for folder in cwd.glob("*") if folder.is_dir()]

    logger.info("Plotting fitted seeing histogram")
    seseeings = []
    for folder in folders:
        hdfstore_path = folder.joinpath("lists.hdf5")

        if hdfstore_path.exists():
            with pd.HDFStore(hdfstore_path, mode='r') as hdfstore:
                if '/calibrated' in hdfstore.keys():
                    calibrated_df = hdfstore.get('/calibrated')
                    seseeings.append(float(calibrated_df['seseeing']))

    plt.hist(seseeings, bins=int(len(seseeings)/4), range=[0.5, 3], color='xkcd:dark grey', histtype='step')
    plt.grid()
    plt.xlabel("Seeing")
    plt.ylabel("Count")
    plt.savefig(cwd.joinpath("{}-{}_seseeing_dist.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

    with open(cwd.joinpath("{}-{}_failures.txt".format(ztfname, filtercode)), 'w') as f:
        # Failure rates
        def _failure_rate(listname, func):
            success_count = 0
            for folder in folders:
                if folder.joinpath("{}.list".format(listname)).exists():
                    success_count += 1

            f.writelines(["For {}:\n".format(func),
                          " Success={}/{}\n".format(success_count, len(folders)),
                          " Rate={}\n\n".format(float(success_count)/len(folders))])

        _failure_rate("se", 'make_catalog')
        _failure_rate("standalone_stars", 'mkcat2')
        _failure_rate("psfstars", 'makepsf')

    logger.info("Plotting computing time histograms")
    # Plot results_*.csv histogram
    results = list(cwd.glob("results_*.csv"))
    for result in results:
        func = str(result.stem).split("_")[1]
        result_df = pd.read_csv(result)
        computation_times = (result_df['time_end'] - result_df['time_start']).to_numpy()
        plt.hist(computation_times, bins=int(len(result_df)/4), histtype='step')
        plt.xlabel("Computation time (s)")
        plt.ylabel("Count")
        plt.title("Computation time for {}".format(result))
        plt.grid()
        plt.savefig(cwd.joinpath("{}-{}_{}_compute_time_dist.png".format(ztfname, filtercode, func)), dpi=300)
        plt.close()


poloka_func.append({'map': stats, 'reduce': stats_reduce})


def smphot(cwd, ztfname, filtercode, logger):
    quadrant_folders = [folder for folder in cwd.glob("ztf_*".format(ztfname, filtercode)) if folder.is_dir()]
    quadrant_folders = list(filter(lambda x: x.joinpath("psfstars.list").exists(), quadrant_folders))

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    seeings = {}

    for quadrant in quadrant_folders:
        calibrated_file = quadrant.joinpath("calibrated.fits")
        with fits.open(calibrated_file) as hdul:
            seeings[quadrant] = (hdul[0].header['seseeing'], hdul[0].header['fieldid'])

    fieldids = list(set([seeing[1] for seeing in seeings.values()]))
    fieldids_count = [sum([1 for f in seeings.values() if f[1]==fieldid]) for fieldid in fieldids]
    maxcount_field = fieldids[np.argmax(fieldids_count)]

    logger.info("{} different field ids".format(len(fieldids)))
    logger.info(fieldids)
    logger.info(fieldids_count)
    logger.info("Max quadrant field={}".format(maxcount_field))

    seeing_df = pd.DataFrame([[quadrant, seeings[quadrant][0]] for quadrant in seeings.keys() if seeings[quadrant][1]==maxcount_field], columns=['quadrant', 'seeing'])
    seeing_df = seeing_df.set_index(['quadrant'])

    # Remove exposure where small amounts of stars are detected
    seeing_df['n_standalonestars'] = list(map(lambda x: len(utils.read_list(pathlib.Path(x).joinpath("standalone_stars.list"))[1]), seeing_df.index))
    seeing_df = seeing_df.loc[seeing_df['n_standalonestars'] >= 25]

    idxmin = seeing_df.idxmin().values[0]
    minseeing = seeing_df.at[idxmin, 'seeing']

    logger.info("Best seeing quadrant: {}". format(idxmin))
    logger.info("  with seeing={}".format(minseeing))

    logger.info("Reading SN1a parameters")
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    logger.info("Reading reference WCS")
    with fits.open(pathlib.Path(idxmin).joinpath("calibrated.fits")) as hdul:
        w = WCS(hdul[0].header)

    ra_px, dec_px = w.world_to_pixel(SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg'))

    logger.info("Writing driver file")
    driver_path = cwd.joinpath("{}_driver_{}".format(ztfname, filtercode))
    logger.info("Writing driver file at location {}".format(driver_path))
    with open(driver_path, 'w') as f:
        f.write("OBJECTS\n")
        f.write("{} {} DATE_MIN={} DATE_MAX={} NAME={} TYPE=0 BAND={}\n".format(ra_px[0],
                                                                                dec_px[0],
                                                                                sn_parameters['t_inf'].values[0],
                                                                                sn_parameters['t_sup'].values[0],
                                                                                ztfname,
                                                                                filtercode))
        f.write("IMAGES\n")
        for quadrant_folder in quadrant_folders:
            f.write("{}\n".format(quadrant_folder))
        f.write("PHOREF\n")
        f.write("{}\n".format(idxmin))
        f.write("PMLIST\n")
        f.write(str(cwd.joinpath("pmfit/pmcatalog.list")))

    # Create GAIA catalog
    logger.info("Building Gaia catalog")

    gaia_cat = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_cat.reset_index(drop=True, inplace=True)

    gaia_cat = gaia_cat.assign(ra_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)
    gaia_cat = gaia_cat.assign(dec_error=pd.Series(np.full(len(gaia_cat), 1e-6)).values)

    gaia_cat = gaia_cat.rename(columns={'pmde': 'pmdec', 'plx': 'parallax', 'e_pmra': 'pmra_error', 'e_pmde': 'pmdec_error', 'gmag': 'g', 'bpmag': 'bp', 'rpmag': 'rp', 'e_gmag': 'g_error', 'e_bpmag': 'bperror', 'e_bpmag': 'bp_error', 'e_rpmag': 'rp_error'})
    gaia_cat = gaia_cat[['ra', 'dec', 'ra_error', 'dec_error', 'pmra', 'pmdec', 'parallax', 'pmra_error', 'pmdec_error', 'g', 'bp', 'rp', 'g_error', 'bp_error', 'rp_error']]

    gaia_path = args.wd.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    np.save(gaia_path, gaia_cat.to_records(index=False))

    logger.info("Running pmfit with polynomial of degree {} as relative astrometric transformation.".format(args.degree))
    run_and_log(["pmfit", driver_path, "-d", str(args.degree), "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--mu-max=20"], logger=logger)

    if args.use_gaia_photom and not filtercode == 'zi':
        # if not cwd.joinpath("stats.csv").exists():
        #     logger.info("Could not find GAIA photometric ratio file... quitting.")
        #     return False

        # logger.info("Retrieving GAIA photometric ratios... (stats.csv)")
        # stats_df = pd.read_csv(cwd.joinpath("stats.csv"))[['quadrant', 'alpha_gaia']].rename(columns={'quadrant': 'expccd', 'alpha_gaia': 'alpha'})
        # stats_df.set_index(stats_df['expccd'], inplace=True)
        # stats_df['ealpha'] = 0
        # shutil.copy(cwd.joinpath("pmfit/photom_ratios.ntuple"), cwd.joinpath("pmfit/photom_ratios.ntuple.ori"))
        # photom_ratios = utils.ListTable.from_filename(cwd.joinpath("pmfit/photom_ratios.ntuple"))
        # photom_ratios.df.set_index(photom_ratios.df['expccd'], inplace=True)
        # photom_ratios.df.loc[stats_df.index] = stats_df
        # photom_ratios.write()

        run_and_log(["gaiafit", "--ztfname={}".format(ztfname), "--filtercode={}".format(filtercode), "--wd={}".format(cwd),
                     "--plots", '--build-measures', "--lc-folder={}".format(args.lc_folder), "--ref-exposure={}".format(idxmin.name)], logger=logger)

    logger.info("Running scene modeling")
    smphot_output = cwd.joinpath("smphot_output")
    smphot_output.mkdir(exist_ok=True)
    run_and_log(["mklc", "-t", cwd.joinpath("pmfit"), "-O", smphot_output, "-v", driver_path], logger=logger)

    return True


poloka_func.append({'reduce': smphot})


def smphot_plot(cwd, ztfname, filtercode, logger):
    logger.info("Running pmfit plots")
    driver_path = cwd.joinpath("{}_driver_{}".format(ztfname, filtercode))
    gaia_path = args.wd.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--plot", "--mu-max=20."], logger=logger)

    logger.info("Running smphot plots")
    with open(cwd.joinpath("smphot_output/lightcurve_sn.dat"), 'r') as f:
        _, sn_flux_df = list_format.read_list(f)

    plt.errorbar(sn_flux_df['mjd'], sn_flux_df['flux'], yerr=sn_flux_df['varflux'], fmt='.k')
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title("Calibrated lightcurve - {} - {}".format(ztfname, filtercode))
    plt.grid()
    plt.savefig(cwd.joinpath("{}-{}_smphot_lightcurve.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

    return True


poloka_func.append({'reduce': smphot_plot})


def match_gaia(quadrant_path, logger):
    if not quadrant_path.joinpath("psfstars.list").exists():
        return

    _, stars_df = utils.read_list(quadrant_path.joinpath("psfstars.list"))
    wcs = utils.get_wcs_from_quadrant(quadrant_path)
    obsmjd = utils.get_mjd_from_quadrant_path(quadrant_path)

    gaia_stars_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_stars_df['gaiaid'] = gaia_stars_df.index

    # Proper motion correction
    gaia_stars_df['ra'] = gaia_stars_df['ra']+(obsmjd-utils.gaiarefmjd)*gaia_stars_df['pmra']/np.cos(gaia_stars_df['dec']/180.*np.pi)/1000./3600/365.
    gaia_stars_df['dec'] = gaia_stars_df['dec']+(obsmjd-utils.gaiarefmjd)*gaia_stars_df['pmde']/1000./3600/365.25

    gaia_stars_radec = SkyCoord(gaia_stars_df['ra'], gaia_stars_df['dec'], unit='deg')
    gaia_mask = utils.contained_in_exposure(gaia_stars_radec, wcs, return_mask=True)
    gaia_stars_df = gaia_stars_df.iloc[gaia_mask]
    x, y = gaia_stars_radec[gaia_mask].to_pixel(wcs)
    gaia_stars_df['x'] = x
    gaia_stars_df['y'] = y

    i = utils.match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), stars_df[['x', 'y']].to_records(), radius=0.5)

    matched_gaia_stars_df = gaia_stars_df.iloc[i[i>=0]].reset_index(drop=True)
    matched_stars_df = stars_df.iloc[i>=0].reset_index(drop=True)
    logger.info("Matched {} GAIA stars".format(len(matched_gaia_stars_df)))

    with pd.HDFStore(quadrant_path.joinpath("matched_stars.hd5"), 'w') as hdfstore:
        hdfstore.put('matched_gaia_stars', matched_gaia_stars_df)
        hdfstore.put('matched_stars', matched_stars_df)


def match_gaia_reduce(cwd, ztfname, filtercode, logger):
    quadrant_paths = [quadrant_path for quadrant_path in list(cwd.glob("ztf_*")) if quadrant_path.is_dir() and quadrant_path.joinpath("psfstars.list").exists()]

    matched_stars_list = []

    for quadrant_path in quadrant_paths:
        matched_gaia_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_gaia_stars')
        matched_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_stars')

        matched_gaia_stars_df.rename(columns={'x': 'gaia_x', 'y': 'gaia_y'}, inplace=True)
        matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['flux'])
        matched_stars_df['emag'] = matched_stars_df['flux']/matched_stars_df['eflux']

        matched_stars_df = pd.concat([matched_stars_df, matched_gaia_stars_df], axis=1)

        header = utils.get_header_from_quadrant_path(quadrant_path)
        matched_stars_df['quadrant'] = quadrant_path.name
        matched_stars_df['airmass'] = header['airmass']
        matched_stars_df['mjd'] = header['obsmjd']
        matched_stars_df['seeing'] = header['seeing']
        matched_stars_df['ha'] = header['hourangd'] #*15
        matched_stars_df['ha_15'] = 15.*header['hourangd']
        matched_stars_df['lst'] = header['oblst']
        matched_stars_df['azimuth'] = header['azimuth']
        matched_stars_df['dome_azimuth'] = header['dome_az']
        matched_stars_df['elevation'] = header['elvation']
        matched_stars_df['z'] = 90. - header['elvation']
        matched_stars_df['telra'] = header['telrad']
        matched_stars_df['teldec'] = header['teldecd']

        matched_stars_list.append(matched_stars_df)

    matched_stars_df = pd.concat(matched_stars_list, axis=0, ignore_index=True)

    # Remove measures with Nan's
    nan_mask = matched_stars_df.isna().any(axis=1)
    matched_stars_df = matched_stars_df[~nan_mask]
    logger.info("Removed {} measurements with Nan's".format(nan_mask))

    # Compute color
    matched_stars_df['colormag'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']

    matched_stars_df.to_parquet(cwd.joinpath("matched_stars.parquet"))
    logger.info("Total matched Gaia stars: {}".format(len(matched_stars_df)))


poloka_func.append({'map': match_gaia, 'reduce': match_gaia_reduce})


def wcs_residuals(cwd, ztfname, filtercode, logger):
    save_folder = cwd.joinpath("res_plots")
    save_folder.mkdir(exist_ok=True)
    matched_stars_df = pd.read_parquet(cwd.joinpath("matched_stars.parquet"))

    ################################################################################
    # Residuals distribution
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.hist(matched_stars_df['x']-matched_stars_df['gaia_x'], bins=100, range=[-0.5, 0.5])
    plt.grid()
    plt.xlabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.ylabel("#")

    plt.subplot(1, 2, 2)
    plt.hist(matched_stars_df['y']-matched_stars_df['gaia_y'], bins=100, range=[-0.5, 0.5])
    plt.grid()
    plt.xlabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.ylabel("#")

    plt.savefig(save_folder.joinpath("wcs_res_dist.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/magnitude
    plt.subplots(nrows=2, ncols=1, figsize=(15., 10.))
    plt.subplot(2, 1, 1)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['x']-matched_stars_df['gaia_x'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmde']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['y']-matched_stars_df['gaia_y'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmde']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.savefig(save_folder.joinpath("mag_wcs_res.png"), dpi=750.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/magnitude binplot
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(matched_stars_df['mag'].to_numpy(), (matched_stars_df['x']-matched_stars_df['gaia_x']).to_numpy(), nbins=50, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{Gaia}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(matched_stars_df['mag'].to_numpy(), (matched_stars_df['y']-matched_stars_df['gaia_y']).to_numpy(), nbins=50, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{Gaia}}$ [pixel]")

    plt.savefig(save_folder.joinpath("mag_wcs_res_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Star lightcurve RMS mag/star lightcurve mean mag
    rms, mean = [], []

    for gaiaid in set(matched_stars_df['gaiaid']):
        gaiaid_mask = (matched_stars_df['gaiaid']==gaiaid)
        rms.append(matched_stars_df.loc[gaiaid_mask, 'mag'].std())
        mean.append(matched_stars_df.loc[gaiaid_mask, 'mag'].mean())

    plt.plot(mean, rms, '.')
    plt.xlabel("$\\left<m\\right>$")
    plt.ylabel("$\\sigma_m$")
    plt.grid()
    plt.savefig(save_folder.joinpath("rms_mean_lc.png"), dpi=300.)
    plt.close()
    ################################################################################


poloka_func.append({'reduce': wcs_residuals})


class AstromModel():
    def __init__(self, dp, degree=5, scale_quadrant=True, quadrant_size=(3072, 3080)):
        self.dp = dp
        self.degree = degree
        self.params = self.init_params()

        self.quadrant_size = quadrant_size
        self.scale = (1./quadrant_size[0], 1./quadrant_size[1])

    def init_params(self):
        self.sky_to_pix = compfuncs.BiPol2D(deg=self.degree, key='quadrant', n=len(self.dp.quadrant_set))
        return fp.FitParameters([*self.sky_to_pix.get_struct(), ('k', 1)])


    @property
    def sigma(self):
        return np.hstack((self.dp.sx, self.dp.sy))

    @property
    def W(self):
        return sparse.dia_array((1./self.sigma**2, 0), shape=(2*len(self.dp.nt), 2*len(self.dp.nt)))

    def __call__(self, x, p, jac=False):
        self.params.free = p
        k = self.params['k'].full
        centered_color = self.dp.color - np.mean(self.dp.color)
        if not jac:
            xy = self.sky_to_pix((self.dp.tpx, self.dp.tpy), p=self.params, quadrant=self.dp.quadrant_index)

            # Could be better implemented
            xy[0] = xy[0] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_x*centered_color
            xy[1] = xy[1] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_y*centered_color

            return xy
        else:
            # Derivatives wrt polynomial
            xy, _, (i, j, vals) = self.sky_to_pix.derivatives(np.array([self.dp.tpx, self.dp.tpy]),
                                                              p=self.params, quadrant=self.dp.quadrant_index)

            # Could be better implemented
            xy[0] = xy[0] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_x*centered_color
            xy[1] = xy[1] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_y*centered_color

            ii = [np.hstack([i, i+len(self.dp.nt)])]
            jj = [np.tile(j, 2).ravel()]
            vv = [np.hstack(vals).ravel()]

            # dm/dk
            i = np.arange(2*len(self.dp.nt))
            ii.append(i)
            jj.append(np.full(2*len(self.dp.nt), self.params['k'].indexof(0)))
            vv.append(np.tile(np.tan(np.deg2rad(self.dp.z)), 2)*np.concatenate([self.dp.parallactic_angle_x, self.dp.parallactic_angle_y])*np.tile(centered_color, 2))

            NN = 2*len(self.dp.nt)
            ii = np.hstack(ii)
            jj = np.hstack(jj)
            vv = np.hstack(vv)
            ok = jj >= 0
            J_model = sparse.coo_array((vv[ok], (ii[ok], jj[ok])), shape=(NN, len(self.params.free)))

            return xy, J_model

    def residuals(self):
        fit_x, fit_y = self.__call__(None, self.params.free)
        return self.dp.x - fit_x, self.dp.y - fit_y


def astrometry_fit(cwd, ztfname, filtercode, logger):
    from sksparse import cholmod
    from imageproc import gnomonic

    # Do the actual astrometry fit
    def fit_astrometry(model):
        print("Astrometry fit with {} measurements.".format(len(model.dp.nt)))
        t = time.perf_counter()
        p = model.params.free.copy()
        v, J = model(None, p, jac=1)
        H = J.T @ model.W @ J
        #H = J.T @ J
        B = J.T @ model.W @ np.hstack((model.dp.x, model.dp.y))
        #B = J.T @ np.hstack((model.dp.x, model.dp.y))
        fact = cholmod.cholesky(H.tocsc())
        p = fact(B)
        model.params.free = p
        print("Done. Elapsed time={}.".format(time.perf_counter()-t))
        return p

    # Filter elements of defined set whose partial Chi2 is over some threshold
    def filter_noisy(model, res_x, res_y, field, threshold):
        w = np.sqrt(res_x**2 + res_y**2)
        field_val = getattr(model.dp, field)
        field_idx = getattr(model.dp, '{}_index'.format(field))
        field_set = getattr(model.dp, '{}_set'.format(field))
        chi2 = np.bincount(field_idx, weights=w)/np.bincount(field_idx)

        noisy = field_set[chi2 > threshold]
        noisy_measurements = np.any([field_val == noisy for noisy in noisy], axis=0)

        model.dp.compress(~noisy_measurements)
        print("Filtered {} {}... down to {} measurements".format(len(noisy), field, len(model.dp.nt)))

        return AstromModel(model.dp, degree=model.degree)

    sn_parameters_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    # Define plot saving folder
    save_folder = cwd.joinpath("astrometry_plots")
    save_folder.mkdir(exist_ok=True)

    # Load data
    matched_stars_df = pd.read_parquet(cwd.joinpath("matched_stars.parquet"))

    # Compute parallactic angle
    parallactic_angle_sin = np.cos(np.deg2rad(utils.ztf_latitude))*np.sin(np.deg2rad(matched_stars_df['ha']))/np.sin(np.deg2rad(matched_stars_df['z']))
    parallactic_angle_cos = np.sqrt(1.-parallactic_angle_sin**2)

    # Project to tangent plane
    tpx, tpy, e_tpx, e_tpy = gnomonic.gnomonic_projection(np.deg2rad(matched_stars_df['ra'].to_numpy()), np.deg2rad(matched_stars_df['dec'].to_numpy()),
                                                          np.deg2rad(sn_parameters_df['sn_ra'].to_numpy()), np.deg2rad(sn_parameters_df['sn_dec'].to_numpy()),
                                                          np.zeros_like(matched_stars_df['ra'].to_numpy()), np.zeros_like(matched_stars_df['dec'].to_numpy()))

    # Add paralactic angle
    matched_stars_df['parallactic_angle_x'] = parallactic_angle_sin
    matched_stars_df['parallactic_angle_y'] = parallactic_angle_cos

    matched_stars_df['tpx'] = tpx[0]
    matched_stars_df['tpy'] = tpy[0]

    plt.plot(tpx[0], tpy[0], '.')
    plt.axis('equal')
    plt.savefig(save_folder.joinpath("tangent_plane_positions.png"), dpi=300.)

    matched_stars_df['color'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']


    # Do cut in magnitude
    matched_stars_df = matched_stars_df.loc[matched_stars_df['mag'] < -10.]

    # Build dataproxy for model
    dp = DataProxy(matched_stars_df.to_records(),
                   x='x', sx='sx', sy='sy', y='y', ra='ra', dec='dec', quadrant='quadrant', mag='mag', gaiaid='gaiaid',
                   bpmag='bpmag', rpmag='rpmag', seeing='seeing', z='z', airmass='airmass', tpx='tpx', tpy='tpy',
                   parallactic_angle_x='parallactic_angle_x', parallactic_angle_y='parallactic_angle_y', color='color', rcid='rcid')

    dp.make_index('quadrant')
    dp.make_index('gaiaid')
    dp.make_index('color')
    dp.make_index('rcid')

    # Build model
    model = AstromModel(dp, degree=7)
    model.init_params()

    # Model fitting
    fit_astrometry(model)
    res_x, res_y = model.residuals()

    # Filter outlier quadrants
    model = filter_noisy(model, res_x, res_y, 'quadrant', 0.1)

    # Redo fit
    fit_astrometry(model)
    res_x, res_y = model.residuals()

    print("k={}".format(model.params['k'].full.item()))

    # Extract and save on disk polynomial coefficients
    coeffs_dict = dict((key, model.params[key].full) for key in model.params._struct.slices.keys())
    coeffs_df = pd.DataFrame(data=coeffs_dict, index=model.dp.quadrant_set)
    coeffs_df.to_csv(save_folder.joinpath("coeffs.csv"), sep=",")

    # Compute partial Chi2 per quadrant and per gaia star
    chi2_quadrant = np.bincount(model.dp.quadrant_index, weights=np.sqrt(res_x**2+res_y**2))/np.bincount(model.dp.quadrant_index)
    chi2_gaiaid = np.bincount(model.dp.gaiaid_index, weights=np.sqrt(res_x**2+res_y**2))/np.bincount(model.dp.gaiaid_index)

    color_mean = np.mean(model.dp.color_set)

    ################################################################################
    # Parallactic angle distribution
    plt.subplot(1, 2, 1)
    plt.hist(matched_stars_df['parallactic_angle_x'], bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("#")

    plt.subplot(1, 2, 1)
    plt.hist(matched_stars_df['parallactic_angle_y'], bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("#")

    plt.savefig(save_folder.joinpath("parallactic_angle_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / quadrant
    # save_folder.joinpath("parallactic_angle_quadrant").mkdir(exist_ok=True)
    # for quadrant in model.dp.quadrant_set:
    #     quadrant_mask = (model.dp.quadrant == quadrant)
    #     plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
    #     plt.subplot(1, 2, 1)
    #     plt.quiver(model.dp.x[quadrant_mask], model.dp.y[quadrant_mask], model.dp.parallactic_angle_x[quadrant_mask], model.dp.parallactic_angle_y[quadrant_mask])
    #     plt.xlim(0., utils.quadrant_width_px)
    #     plt.ylim(0., utils.quadrant_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.subplot(1, 2, 2)
    #     plt.quiver(model.dp.x[quadrant_mask], model.dp.y[quadrant_mask], res_x[quadrant_mask], res_y[quadrant_mask])
    #     plt.xlim(0., utils.quadrant_width_px)
    #     plt.ylim(0., utils.quadrant_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.savefig(save_folder.joinpath("parallactic_angle_quadrant/parallactic_angle_{}.png".format(quadrant)), dpi=150.)
    #     plt.close()

    ################################################################################
    # Color distribution
    plt.hist(model.dp.color_set-color_mean, bins=25)
    plt.xlabel("$B_p-R_p-\\left<B_p-R_p\\right>$ [mag]")

    plt.savefig(save_folder.joinpath("color_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Athmospheric refraction / residuals
    plt.subplots(ncols=1, nrows=2, figsize=(20., 10.))
    plt.subplot(2, 1, 1)
    plt.plot(np.tan(np.deg2rad(model.dp.z))*model.dp.parallactic_angle_x*(model.dp.color-color_mean), res_x, ',')
    # idx2marker = {0: '*', 1: '.', 2: 'o', 3: 'x'}
    # for i, rcid in enumerate(model.dp.rcid_set):
    #     rcid_mask = (model.dp.rcid == rcid)
    #     plt.scatter(np.tan(np.deg2rad(model.dp.z[rcid_mask]))*model.dp.parallactic_angle_x[rcid_mask][:, 0]*(model.dp.color[rcid_mask]-color_mean), res_x[rcid_mask], marker=idx2marker[i], label=rcid, s=0.1)

    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\sin(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$x-x_\\mathrm{fit}$")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.tan(np.deg2rad(model.dp.z))*model.dp.parallactic_angle_y*(model.dp.color-color_mean), res_y, ',')
    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\cos(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$y-y_\\mathrm{fit}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("atmref_residuals.pdf"), dpi=300.)
    plt.close()

    ################################################################################
    # Chi2/quadrant / seeing
    plt.plot(model.dp.seeing, chi2_quadrant[model.dp.quadrant_index], '.')
    plt.xlabel("Seeing")
    plt.ylabel("$\\chi^2_\\mathrm{quadrant}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("chi2_quadrant_seeing.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2/quadrant / airmass
    plt.plot(model.dp.airmass, chi2_quadrant[model.dp.quadrant_index], '.')
    plt.xlabel("Airmass")
    plt.ylabel("$\\chi^2_\\mathrm{quadrant}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("chi2_quadrant_airmass.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / distance to origin
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(model.dp.x**2+model.dp.y**2), res_x, ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$x-x_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(model.dp.x**2+model.dp.y**2), res_y, ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$y-y_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.savefig(save_folder.joinpath("residuals_origindistance.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / star index
    plt.plot(range(len(model.dp.gaiaid_set)), chi2_gaiaid, ".", color='black')
    plt.xlabel("Gaia #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(save_folder.joinpath("chi2_star.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / quadrant index
    plt.plot(range(len(model.dp.quadrant_set)), chi2_quadrant, ".", color='black')
    plt.xlabel("Quadrant #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(save_folder.joinpath("chi2_quadrant.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals distribution
    plt.subplot(1, 2, 1)
    plt.hist(res_x, bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(1, 2, 2)
    plt.hist(res_y, bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("residuals_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Magnitude / residuals
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(model.dp.mag, res_x, ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(model.dp.mag, res_y, ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("magnitude_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    plt.subplots(nrows=2, ncols=2, figsize=(10., 10.))
    # Magnitude / residuals binplot
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, res_x, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, res_y, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{git}}$ [pixel]")

    plt.savefig(save_folder.joinpath("magnitude_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color plot
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(model.dp.bpmag-model.dp.rpmag, res_x, ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(model.dp.bpmag-model.dp.rpmag, res_y, ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("color_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color binplot
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(model.dp.bpmag-model.dp.rpmag, res_x, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(model.dp.bpmag-model.dp.rpmag, res_y, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.savefig(save_folder.joinpath("color_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################


poloka_func.append({'reduce': astrometry_fit})


poloka_func = dict(zip([list(func.values())[0].__name__ for func in poloka_func], poloka_func))


def dump_timings(start_time, end_time, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps({'start': start_time, 'end': end_time, 'elapsed': end_time-start_time}))


scratch_files_to_ignore = ["output.log"]
def map_op(quadrant, wd, ztfname, filtercode, func, scratch=None):
    start_time = time.perf_counter()
    quadrant_dir = wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))

    logger = None
    if func != 'clean':
        logger = logging.getLogger(quadrant)
        logger.addHandler(logging.FileHandler(str(quadrant_dir.joinpath("output.log")), mode='a'))
        logger.setLevel(logging.INFO)
        logger.info(datetime.datetime.today())
        logger.info("Current directory: {}".format(quadrant_dir))
        logger.info("Running {}".format(func))

        logger.info("Quadrant directory: {}".format(quadrant_dir))

        if scratch:
            logger.info("Using scratch space {}".format(scratch))
            logger.info(" Parent exists={}".format(scratch.parent.exists()))
            quadrant_scratch = scratch.joinpath(quadrant)
            quadrant_scratch.mkdir(exist_ok=True, parents=True)
            logger.info("Successfully created quadrant working dir in scratch space")
            files = list(quadrant_dir.glob("*"))

            [shutil.copyfile(f, quadrant_scratch.joinpath(f.name)) for f in files if f.name not in scratch_files_to_ignore]
            quadrant_dir = quadrant_scratch
            logger.info("Successfully copyed files from sps to scratchspace")

    result = False
    try:
        start_time = time.perf_counter()
        result = poloka_func[func]['map'](quadrant_dir, logger)
    except Exception as e:
        logger.error("")
        logger.error("In folder {}".format(quadrant_dir))
        logger.error(traceback.format_exc())
        print(traceback.format_exc())
    finally:
        end_time = time.perf_counter()
        if args.dump_timings:
            dump_timings(start_time, end_time, quadrant_dir.joinpath("timings_{}".format(func)))
        if scratch and func != 'clean':
            logger.info("Erasing quadrant data from scratchspace")
            files = list(quadrant_dir.glob("*"))
            [shutil.copy2(f, wd.joinpath("{}/{}/{}".format(ztfname, filtercode, quadrant))) for f in files]
            [f.unlink() for f in files]
            quadrant_dir.rmdir()

    if func != 'clean':
        logger.info("Done.")

    return result, time.perf_counter(), start_time, get_worker().id


def reduce_op(results, cwd, ztfname, filtercode, func, save_stats):
    folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))

    # If we want to agregate run statistics on the previous map operation
    if save_stats and results is not None and any(results) and func != 'clean':
        results_df = pd.DataFrame([result for result in results if result is not None], columns=['result', 'time_end', 'time_start', 'worker_id'])
        results_df.to_csv(folder.joinpath("results_{}.csv".format(func)), index=False)

    if not 'reduce' in poloka_func[func].keys():
        return

    logger = logging.getLogger("{}-{}".format(ztfname, filtercode))
    logger.addHandler(logging.FileHandler(folder.joinpath("output.log"), mode='a'))
    logger.setLevel(logging.INFO)
    logger.info(datetime.datetime.today())
    logger.info("Running reduction {}".format(args.func))

    start_time = time.perf_counter()
    try:
        result = poloka_func[func]['reduce'](folder, ztfname, filtercode, logger)
    except Exception as e:
        logger.error("")
        logger.error("In SN {}-{}".format(ztfname, filtercode))
        logger.error(traceback.format_exc())
        print(traceback.format_exc())
    finally:
        pass

    end_time = time.perf_counter()
    if args.dump_timings:
        dump_timings(start_time, end_time, folder.joinpath("timings_{}".format(func)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--ztfname', type=pathlib.Path, help="If provided, perform computation on one SN1a. If it points to a valid text file, will perform computation on all keys. If not provided, process the whole working directory.")
    argparser.add_argument('-j', dest='n_jobs', type=int, default=1)
    argparser.add_argument('--wd', type=pathlib.Path, help="Working directory")
    argparser.add_argument('--filtercode', choices=ztf_filtercodes, default='all', help="Only perform computations on one or all filters.")
    argparser.add_argument('--func', type=str, choices=poloka_func.keys(), default='pipeline')
    argparser.add_argument('--dry-run', dest='dry_run', action='store_true')
    argparser.add_argument('--no-map', dest='no_map', action='store_true')
    argparser.add_argument('--no-reduce', dest='no_reduce', action='store_true')
    argparser.add_argument('--cluster-worker', type=int, default=0)
    argparser.add_argument('--scratch', type=pathlib.Path)
    argparser.add_argument('--lc-folder', dest='lc_folder', type=pathlib.Path)
    argparser.add_argument('--log-results', action='store_true', default=True)
    argparser.add_argument('--degree', type=int, default=3, help="Degree of polynomial for relative astrometric fit in pmfit.")
    argparser.add_argument('--use-gaia-photom', action='store_true', help="Use photometric ratios computed using GAIA")
    argparser.add_argument('--dump-timings', action='store_true')

    args = argparser.parse_args()
    args.wd = args.wd.expanduser().resolve()


    filtercodes = ztf_filtercodes[:3]
    if args.filtercode != 'all':
        filtercodes = [args.filtercode]


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

    if args.scratch:
        args.scratch.mkdir(exist_ok=True, parents=True)

        import signal
        import atexit
        def delete_scratch_at_exit(scratch_dir):
            shutil.rmtree(scratch_dir)

        atexit.register(delete_scratch_at_exit, scratch_dir=args.scratch)


    if args.cluster_worker > 0:
        cluster = SLURMCluster(cores=args.n_jobs,
                               processes=args.n_jobs,
                               memory="{}GB".format(3*n_jobs),
                               project="ztf",
                               walltime="12:00:00",
                               queue="htc",
                               job_extra=["-L sps"])

        cluster.scale(jobs=args.cluster_worker)
        client = Client(cluster)
        print(client.dashboard_link, flush=True)
        print(socket.gethostname(), flush=True)
        print("Running {} workers with {} processes each ({} total).".format(args.cluster_worker, args.n_jobs, args.cluster_worker*args.n_jobs))
        client.wait_for_workers(1)
    else:
        if args.n_jobs == 1:
            dask.config.set(scheduler='synchronous')

        localCluster = LocalCluster(n_workers=args.n_jobs, dashboard_address='localhost:8787', threads_per_worker=1, nanny=False)
        client = Client(localCluster)
        print("Dask dashboard at: {}".format(client.dashboard_link))


    jobs = []
    quadrant_count = 0
    reduction_count = 0
    for ztfname in ztfnames:
        for filtercode in filtercodes:
            print("Building job list for {}-{}... ".format(ztfname, filtercode), end="", flush=True)

            quadrants_folder = args.wd.joinpath("{}/{}".format(ztfname, filtercode))
            if not quadrants_folder.exists():
                print("No quadrant found.")
                continue

            results = None

            if 'map' in poloka_func[args.func].keys() and not args.no_map:
                quadrants = list(map(lambda x: x.stem, filter(lambda x: x.is_dir(), quadrants_folder.glob("ztf*"))))
                quadrant_count += len(quadrants)

                results = [delayed(map_op)(quadrant, args.wd, ztfname, filtercode, args.func, args.scratch) for quadrant in quadrants]
                print("Found {} quadrants. ".format(len(quadrants)), end="", flush=True)

            if ('reduce' in poloka_func[args.func].keys() or args.log_results) and not args.no_reduce:
                results = [delayed(reduce_op)(results, args.wd, ztfname, filtercode, args.func, True)]
                reduction_count += 1
                print("Found reduction.", end="", flush=True)

            print("")

            if results:
                jobs.extend(results)

    print("")
    print("Running. ", end="")

    if quadrant_count > 0:
        print(" Processing {} quadrants.".format(quadrant_count))

    if reduction_count > 0:
        print(" Processing {} reductions.".format(reduction_count))

    start_time = time.perf_counter()
    fjobs = client.compute(jobs)
    wait(fjobs)
    print("Done. Elapsed time={}".format(time.perf_counter() - start_time))

    client.close()
