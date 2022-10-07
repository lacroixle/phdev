#!/usr/bin/env python3


def psf_study(quadrant_path, logger, args):
    from utils import ListTable, match_pixel_space
    import numpy as np
    from saunerie.plottools import binplot
    from scipy.stats import norm
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    if not quadrant_path.joinpath("psfstars.list").exists():
        return True

    psf_resid = ListTable.from_filename(quadrant_path.joinpath("psf_resid_tuple.dat"))
    psf_tuple = ListTable.from_filename(quadrant_path.joinpath("psftuple.list"))
    psf_stars = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))
    stand = ListTable.from_filename(quadrant_path.joinpath("standalone_stars.list"))

    stand.df['mag'] = -2.5*np.log10(stand.df['flux'])
    psf_stars.df['mag'] = -2.5*np.log10(psf_stars.df['flux'])
    psf_residuals_px = (psf_resid.df['fimg'] - psf_resid.df['fpsf']).to_numpy()


    plt.figure(figsize=(7., 7.))
    plt.suptitle("Skewness plane for standalone stars stamps in \n {}".format(quadrant_path.name))
    plt.scatter(stand.df['gmx3'], stand.df['gmy3'], s=2., color='black')
    plt.xlabel("$M^g_{xxx}$")
    plt.ylabel("$M^g_{yyy}$")
    plt.axvline(0.)
    plt.axhline(0.)
    plt.axis('equal')
    plt.grid()
    plt.savefig(quadrant_path.joinpath("{}_mx3_my3_plane.png".format(quadrant_path.name)), dpi=300.)
    plt.close()

    plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(10., 5.))
    plt.suptitle("Skewness vs magnitude for standalone stars stamps in {}".format(quadrant_path.name))
    plt.subplot(2, 1, 1)
    plt.scatter(stand.df['mag'], stand.df['gmx3'], s=2.)
    plt.axhline(0.)
    plt.grid()
    plt.ylim(-0.5, 0.3)
    plt.xlim(-15., -7)
    plt.ylabel("$M^g_{xxx}$")

    plt.subplot(2, 1, 2)
    plt.scatter(stand.df['mag'], stand.df['gmy3'], s=2.)
    plt.axhline(0.)
    plt.xlabel("$m$")
    plt.grid()
    plt.ylim(-0.3, 0.3)
    plt.xlim(-15., -6)
    plt.ylabel("$M^g_{yyy}$")

    plt.savefig(quadrant_path.joinpath("{}_skewness_magnitude.png".format(quadrant_path.name)), dpi=300.)
    plt.close()

    star_indices = match_pixel_space(psf_stars.df[['x', 'y']].to_records(), psf_resid.df[['xc', 'yc']].rename(mapper={'xc': 'x', 'yc': 'y'}, axis='columns').to_records())
    psf_size = int(np.sqrt(sum(star_indices==0)))
    psf_residuals = np.zeros([len(psf_stars.df), psf_size*psf_size])
    for star_index in range(len(np.bincount(star_indices))):
        star_mask = (star_indices==star_index)
        ij = (psf_resid.df.iloc[star_mask]['j']*psf_size + psf_resid.df.iloc[star_mask]['i'] + int(np.floor(psf_size**2/2))).to_numpy()
        np.put_along_axis(psf_residuals[star_index], ij, psf_residuals_px[star_mask], 0)

    psf_residuals = psf_residuals.reshape(len(psf_stars.df), psf_size, psf_size)

    bins_count = 5
    mag_range = (psf_stars.df['mag'].min(), psf_stars.df['mag'].max())
    bins = np.linspace(*mag_range, bins_count+1)
    psf_residual_means = []
    psf_residual_count = []
    for i in range(bins_count):
        lower_bound = bins[i]
        upper_bound = bins[i+1]

        binned_stars_mask = np.all([psf_stars.df['mag'] < upper_bound, psf_stars.df['mag'] >= lower_bound], axis=0)
        psf_residual_means.append(np.mean(psf_residuals[binned_stars_mask], axis=0))
        psf_residual_count.append(sum(binned_stars_mask))

    plt.subplots(ncols=bins_count, nrows=1, figsize=(5.*bins_count, 5.))
    plt.suptitle("Binned PSF residuals for {}".format(quadrant_path.name))
    for i, psf_residual_mean in enumerate(psf_residual_means):
        plt.subplot(1, bins_count,  1+i)
        plt.imshow(psf_residual_mean)
        plt.axis('off')
        plt.title("${0:.2f} \leq m < {1:.2f}$\n$N={2}$".format(bins[i], bins[i+1], psf_residual_count[i]))

    plt.savefig(quadrant_path.joinpath("{}_psf_residuals.png".format(quadrant_path.name)))
    plt.close()
    # plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
    # # plt.subplot(1, 2, 1)
    # # plt.scatter(stand.df['gmx3'], stand.df['gmy3'], c=stand.df['x'], s=2.)
    # # plt.axvline(0.)
    # # plt.axhline(0.)

    # # plt.subplot(1, 2, 2)
    # # plt.scatter(stand.df['gmx3'], stand.df['gmy3'], c=stand.df['y'], s=2.)
    # # plt.axvline(0.)
    # # plt.axhline(0.)
    # # plt.show()

    # plt.subplot(2, 2, 1)
    # plt.scatter(stand.df['mag'], stand.df['gmx3'], c=stand.df['x'], s=1.)
    # plt.axhline(0.)
    # plt.subplot(2, 2, 2)
    # plt.scatter(stand.df['mag'], stand.df['gmy3'], c=stand.df['x'])
    # plt.axhline(0.)
    # plt.subplot(2, 2, 3)
    # plt.scatter(stand.df['mag'], stand.df['gmx3'], c=stand.df['y'])
    # plt.axhline(0.)
    # plt.subplot(2, 2, 4)
    # plt.scatter(stand.df['mag'], stand.df['gmy3'], c=stand.df['y'])
    # plt.axhline(0.)
    # plt.show()

    # psf_resid.df.to_csv("psf_resid_tuple.csv", sep=",")
    # psf_tuple.df.to_csv("psftuple.csv", sep=",")
    # psf_stars.df.to_csv("psfstars.csv", sep=",")


    # psf_resid.df['flux'] = psf_stars.df['flux'].iloc[idx].reset_index(drop=True)
    # psf_resid.df['eflux'] = psf_stars.df['eflux'].iloc[idx].reset_index(drop=True)



    # limits = [np.min(res), np.max(res)]
    # f = np.linspace(*limits, 200)
    # s = np.var(res)
    # m = np.mean(res)
    # print(np.sqrt(s))
    # print(m)

    # plt.hist(res, bins=1000, density=True)
    # plt.plot(f, norm.pdf(f, loc=m, scale=np.sqrt(s)))
    # plt.xlim(limits)
    # plt.show()

    # plt.plot(psf_resid.df['flux'], psf_resid.df['eflux']/psf_resid.df['flux'], '.')
    # plt.show()

    # plt.subplots(nrows=2, ncols=1, figsize=(10., 5.), sharex=True)
    # plt.subplot(2, 1, 1)
    # binned_mag, plot_res, res_dispersion = binplot(psf_resid.df['flux'].to_numpy(), psf_resid.df['fimg'] - psf_resid.df['fpsf'], nbins=10, data=True, rms=True, scale=False)

    # plt.ylabel("res")
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(binned_mag, res_dispersion, color='black')
    # plt.xlabel("Flux [ADU]")
    # plt.ylabel("$\sigma_\\mathrm{res}$")
    # plt.grid()

    # plt.show()

    return True


def psf_study_reduce(band_path, ztfname, filtercode, logger, args):

    pass


def match_gaia(quadrant_path, logger, args):
    from utils import read_list, get_wcs_from_quadrant, get_mjd_from_quadrant_path, contained_in_exposure, match_pixel_space, gaiarefmjd
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord

    ztfname = quadrant_path.parts[-3]

    if not quadrant_path.joinpath("psfstars.list").exists():
        return

    _, stars_df = read_list(quadrant_path.joinpath("psfstars.list"))
    wcs = get_wcs_from_quadrant(quadrant_path)
    obsmjd = get_mjd_from_quadrant_path(quadrant_path)

    gaia_stars_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['gaiaid'] = gaia_stars_df.index

    # First order proper motion correction for indentification
    # We do it in astrometry code
    # gaia_stars_df['ra'] = gaia_stars_df['ra']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmra']/np.cos(gaia_stars_df['dec']/180.*np.pi)/1000./3600./365.25
    # gaia_stars_df['dec'] = gaia_stars_df['dec']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmde']/1000./3600./365.25
    # mas/year -> deg/day
    gaia_stars_df['pmra'] = gaia_stars_df['pmra']/np.cos(np.deg2rad(gaia_stars_df['dec']))/1000./3600./365.25
    gaia_stars_df['pmdec'] = gaia_stars_df['pmdec']/1000./3600./365.25

    ra_corrected = gaia_stars_df['ra']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmra']
    dec_corrected = gaia_stars_df['dec']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmdec']

    # gaia_stars_radec = SkyCoord(gaia_stars_df['ra'], gaia_stars_df['dec'], unit='deg')
    gaia_stars_radec = SkyCoord(ra_corrected, dec_corrected, unit='deg')
    gaia_mask = contained_in_exposure(gaia_stars_radec, wcs, return_mask=True)
    gaia_stars_df = gaia_stars_df.iloc[gaia_mask]
    x, y = gaia_stars_radec[gaia_mask].to_pixel(wcs)
    gaia_stars_df['x'] = x
    gaia_stars_df['y'] = y

    i = match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), stars_df[['x', 'y']].to_records(), radius=0.5)

    matched_gaia_stars_df = gaia_stars_df.iloc[i[i>=0]].reset_index(drop=True)
    matched_stars_df = stars_df.iloc[i>=0].reset_index(drop=True)
    logger.info("Matched {} GAIA stars".format(len(matched_gaia_stars_df)))

    with pd.HDFStore(quadrant_path.joinpath("matched_stars.hd5"), 'w') as hdfstore:
        hdfstore.put('matched_gaia_stars', matched_gaia_stars_df)
        hdfstore.put('matched_stars', matched_stars_df)


def match_gaia_reduce(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    import numpy as np
    from utils import get_header_from_quadrant_path

    quadrant_paths = [quadrant_path for quadrant_path in list(band_path.glob("ztf_*")) if quadrant_path.is_dir() and quadrant_path.joinpath("psfstars.list").exists()]

    matched_stars_list = []
    quadrants_dict = {}

    for quadrant_path in quadrant_paths:
        matched_gaia_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_gaia_stars')
        matched_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_stars')

        matched_gaia_stars_df.rename(columns={'x': 'gaia_x', 'y': 'gaia_y'}, inplace=True)
        matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['flux'])
        matched_stars_df['emag'] = matched_stars_df['eflux']/matched_stars_df['flux']

        matched_stars_df = pd.concat([matched_stars_df, matched_gaia_stars_df], axis=1)

        quadrant_dict = {}
        header = get_header_from_quadrant_path(quadrant_path)
        quadrant_dict['quadrant'] = quadrant_path.name
        quadrant_dict['airmass'] = header['airmass']
        quadrant_dict['mjd'] = header['obsmjd']
        quadrant_dict['seeing'] = header['seeing']
        quadrant_dict['ha'] = header['hourangd'] #*15
        quadrant_dict['ha_15'] = 15.*header['hourangd']
        quadrant_dict['lst'] = header['oblst']
        quadrant_dict['azimuth'] = header['azimuth']
        quadrant_dict['dome_azimuth'] = header['dome_az']
        quadrant_dict['elevation'] = header['elvation']
        quadrant_dict['z'] = 90. - header['elvation']
        quadrant_dict['telra'] = header['telrad']
        quadrant_dict['teldec'] = header['teldecd']
        quadrant_dict['rcid'] = header['rcid']
        quadrant_dict['temperature'] = header['tempture']
        quadrant_dict['head_temperature'] = header['headtemp']
        quadrant_dict['wind_speed'] = header['windspd']
        quadrant_dict['wind_dir'] = header['winddir']
        quadrant_dict['dewpoint'] = header['dewpoint']

        for key in quadrant_dict.keys():
            matched_stars_df[key] = quadrant_dict[key]

        matched_stars_list.append(matched_stars_df)
        quadrants_dict[quadrant_path.name] = quadrant_dict

    matched_stars_df = pd.concat(matched_stars_list, axis=0, ignore_index=True)

    # Build gaia star catalog
    gaia_stars_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['gaiaid'] = gaia_stars_df.index
    gaia_stars = []
    for gaiaid in set(matched_stars_df['gaiaid']):
        #gaia_stars.append({gaiaid: gaia_stars_df.loc[gaiaid].to_dict()})
        gaia_stars.append(pd.Series(gaia_stars_df.loc[gaiaid], name=gaiaid))

    gaia_stars_df = pd.DataFrame(data=gaia_stars)
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['pmra'] = gaia_stars_df['pmra']/np.cos(np.deg2rad(gaia_stars_df['dec']))/1000./3600./365.25
    gaia_stars_df['pmdec'] = gaia_stars_df['pmdec']/1000./3600./365.25

    # Remove measures with Nan's
    nan_mask = matched_stars_df.isna().any(axis=1)
    matched_stars_df = matched_stars_df[~nan_mask]
    logger.info("Removed {} measurements with NaN's".format(sum(nan_mask)))

    nan_mask = gaia_stars_df.isna().any(axis=1)
    gaia_stars_df = gaia_stars_df[~nan_mask]
    logger.info("Removed {} Gaia stars with NaN's".format(sum(nan_mask)))

    # Compute color
    matched_stars_df['colormag'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']
    gaia_stars_df['colormag'] = gaia_stars_df['bpmag'] - gaia_stars_df['rpmag']

    # Save to disk
    matched_stars_df.to_parquet(band_path.joinpath("matched_stars.parquet"))
    logger.info("Total matched Gaia stars: {}".format(len(matched_stars_df)))

    quadrants_df = pd.DataFrame.from_dict(quadrants_dict, orient='index')
    quadrants_df.to_parquet(band_path.joinpath("quadrants.parquet"))

    gaia_stars_df.to_parquet(band_path.joinpath("gaia_stars.parquet"))


# Extract data from standalone stars and plot several distributions
def stats(quadrant_path, logger, args):
    import warnings
    import pandas as pd
    from utils import read_list
    from astropy.io import fits

    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

    import pandas as pd

    def _extract_from_list(list_filename, hdfstore):
        list_path = quadrant_path.joinpath(list_filename).with_suffix(".list")

        if not list_path.exists():
            return False

        with open(list_path, mode='r') as f:
            global_params, df = read_list(f)

        hdfstore.put(list_path.stem, df)
        hdfstore.put("{}_globals".format(list_path.stem), pd.DataFrame([global_params]))

        return True

    with pd.HDFStore(quadrant_path.joinpath("lists.hdf5"), mode='w') as hdfstore:
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
        with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
            for keyword in keywords:
                calibrated[keyword] = hdul[0].header[keyword]

            hdfstore.put('calibrated', pd.DataFrame([calibrated]))

        # From makepsf
        cont = _extract_from_list("psfstars", hdfstore)

        if not cont:
            return True

        _extract_from_list("psftuples", hdfstore)

    return True


def stats_reduce(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Seeing histogram
    folders = [folder for folder in band_path.glob("*") if folder.is_dir()]

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
    plt.savefig(band_path.joinpath("{}-{}_seseeing_dist.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

    with open(band_path.joinpath("{}-{}_failures.txt".format(ztfname, filtercode)), 'w') as f:
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
    result_paths = list(band_path.glob("results_*.csv"))
    for result_path in result_paths:
        func = "_".join(str(result_path.stem).split("_")[1:])
        result_df = pd.read_csv(result_path)
        computation_times = (result_df['time_end'] - result_df['time_start']).to_numpy()
        plt.hist(computation_times, bins=int(len(result_df)/4), histtype='step')
        plt.xlabel("Computation time (s)")
        plt.ylabel("Count")
        plt.title("Computation time for {}".format(func))
        plt.grid()
        plt.savefig(band_path.joinpath("{}-{}_{}_compute_time_dist.png".format(ztfname, filtercode, func)), dpi=300)
        plt.close()


def clean(quadrant_path, logger, args):
    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["elixir.fits", "dead.fits.gz", ".dbstuff"]
    files_to_delete = list(filter(lambda f: f.name not in files_to_keep, list(quadrant_path.glob("*"))))

    for file_to_delete in files_to_delete:
            file_to_delete.unlink()

    return True


def clean_reduce(band_path, ztfname, filtercode, logger, args):
    from shutil import rmtree

    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["prepare.log"]

    # Delete all files
    files_to_delete = list(filter(lambda f: f.is_file() and (f.name not in files_to_keep), list(band_path.glob("*"))))

    [f.unlink() for f in files_to_delete]

    # Delete output folders
    rmtree(band_path.joinpath("pmfit"), ignore_errors=True)
    rmtree(band_path.joinpath("pmfit_plot"), ignore_errors=True)
    rmtree(band_path.joinpath("smphot_output"), ignore_errors=True)
    rmtree(band_path.joinpath("wcs_residuals_plots"), ignore_errors=True)
