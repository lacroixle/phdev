#!/usr/bin/env python3


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
    gaia_stars_df['gaiaid'] = gaia_stars_df.index

    # Proper motion correction
    gaia_stars_df['ra'] = gaia_stars_df['ra']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmra']/np.cos(gaia_stars_df['dec']/180.*np.pi)/1000./3600./365.25
    gaia_stars_df['dec'] = gaia_stars_df['dec']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmde']/1000./3600./365.25

    gaia_stars_radec = SkyCoord(gaia_stars_df['ra'], gaia_stars_df['dec'], unit='deg')
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

    # Remove measures with Nan's
    nan_mask = matched_stars_df.isna().any(axis=1)
    matched_stars_df = matched_stars_df[~nan_mask]
    logger.info("Removed {} measurements with Nan's".format(nan_mask))

    # Compute color
    matched_stars_df['colormag'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']

    # Save to disk
    matched_stars_df.to_parquet(band_path.joinpath("matched_stars.parquet"))
    logger.info("Total matched Gaia stars: {}".format(len(matched_stars_df)))

    quadrants_df = pd.DataFrame.from_dict(quadrants_dict, orient='index')
    quadrants_df.to_parquet(band_path.joinpath("quadrants.parquet"))


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

    #files_to_delete = [file_to_delete for file_to_delete in files if file_to_delete.name not in files_to_keep]
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
