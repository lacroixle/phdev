#!/usr/bin/env python3

from deppol_utils import run_and_log


def reference_quadrant(lightcurve, logger, args):
    """

    """
    # TODO: Rewrite this using quadrants.parquet instead of reading all fits headers
    import pathlib
    import pandas as pd
    import numpy as np

    from deppol_utils import update_yaml

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    exposures = lightcurve.get_exposures(files_to_check="psfstars.list")
    seeings = dict([(exposure.name, (exposure.exposure_header['seeing'], exposure.field)) for exposure in exposures])
    seeings_df = pd.DataFrame({'seeing': list(map(lambda x: x.exposure_header['seeing'], exposures)),
                               'fieldid': list(map(lambda x: x.field, exposures)),
                               'star_count': list(map(lambda x: len(x.get_matched_catalog('psfstars')), exposures))},
                              index=list(map(lambda x: x.name, exposures)))

    fieldids = seeings_df.value_counts(subset='fieldid').index.tolist()
    fieldids_count = seeings_df.value_counts(subset='fieldid').values
    maxcount_fieldid = fieldids[0]

    logger.info("Computing reference exposure from {} exposures".format(len(exposures)))
    logger.info("{} different field ids".format(len(fieldids)))
    logger.info("Field ids: {}".format(fieldids))
    logger.info("Count: {}".format(fieldids_count))
    logger.info("Max quadrant field={}".format(maxcount_fieldid))
    logger.info("Determining the minimum number of stars the reference must have")
    if args.min_psfstars:
        logger.info("  --min-psfstars defined.")
        min_psfstars = args.min_psfstars
    else:
        logger.info("  --min-psfstars not defined, use 4(N+1)(N+2) with N the relative astrometry polynomial degree.")
        min_psfstars = 4*(args.astro_degree+1)*(args.astro_degree+2)
    logger.info("Minimum stars for reference={}".format(min_psfstars))

    idxmin = seeings_df.loc[seeings_df['fieldid']==maxcount_fieldid].loc[seeings_df['star_count']>=min_psfstars]['seeing'].idxmin()
    minseeing = seeings_df.loc[idxmin, 'seeing']

    logger.info("Best seeing quadrant: {}". format(idxmin))
    logger.info("  with seeing={}".format(minseeing))
    logger.info("  psfstars count={}".format(len(lightcurve.exposures[idxmin].get_catalog('psfstars.list').df)))

    logger.info("Writing into {}".format(lightcurve.path.joinpath("reference_exposure")))
    with open(lightcurve.path.joinpath("reference_exposure"), 'w') as f:
        f.write(idxmin)

    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'reference',
                {'name': idxmin,
                 'seeing': minseeing.item(),
                 'psfstarcount': len(lightcurve.exposures[idxmin].get_catalog('psfstars.list').df),
                 'mjd': float(lightcurve.exposures[idxmin].exposure_header['obsmjd']),
                 'min_psfstars': float(min_psfstars),
                 'fieldids': fieldids,
                 'fieldids_count': fieldids_count.tolist(),
                 'maxcount_fieldid': maxcount_fieldid})

    return True


def smphot(lightcurve, logger, args):
    import numpy as np
    import pandas as pd
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    from utils import ListTable

    logger.info("Writing driver file")
    exposures = lightcurve.get_exposures(files_to_check='psfstars.list')
    reference_exposure = lightcurve.get_reference_exposure()

    logger.info("Reading SN1a parameters")
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')

    ra_px, dec_px = lightcurve.exposures[reference_exposure].wcs.world_to_pixel(SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg'))

    logger.info("Writing driver file")
    driver_path = lightcurve.path.joinpath("smphot_driver")
    logger.info("Writing driver file at location {}".format(driver_path))
    with open(driver_path, 'w') as f:
        f.write("OBJECTS\n")
        f.write("{} {} DATE_MIN={} DATE_MAX={} NAME={} TYPE=0 BAND={}\n".format(ra_px[0],
                                                                                dec_px[0],
                                                                                sn_parameters['t_inf'].values[0],
                                                                                sn_parameters['t_sup'].values[0],
                                                                                lightcurve.name,
                                                                                lightcurve.filterid[1]))
        f.write("IMAGES\n")
        # for exposure in exposures:
        #     f.write("{}\n".format(exposure.path))
        [f.write("{}\n".format(exposure.path)) for exposure in exposures if exposure.name != reference_exposure]
        f.write("PHOREF\n")
        f.write("{}\n".format(str(lightcurve.path.joinpath(reference_exposure))))
        f.write("PMLIST\n")
        f.write(str(lightcurve.path.joinpath("astrometry/pmcatalog.list")))


    logger.info("Running scene modeling")

    lightcurve.smphot_path.mkdir(exist_ok=True)

    returncode = run_and_log(["mklc", "-t", lightcurve.mappings_path, "-O", lightcurve.smphot_path, "-v", lightcurve.driver_path], logger=logger)

    logger.info("Deleting unuseful *.fits files...")
    to_delete_list = list(lightcurve.smphot_path.glob("*.fits"))
    if returncode == 0:
        sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')
        fit_df = ListTable.from_filename(lightcurve.smphot_path.joinpath("lc2fit.dat")).df
        t0 = sn_parameters['t0mjd'].item()
        t0_idx = np.argmin(np.abs(fit_df['Date']-t0))
        t0_exposure = fit_df.iloc[t0_idx]['name']
        print("Keeping t0 image {}".format(t0_exposure))
        to_delete_list.remove(lightcurve.smphot_path.joinpath(t0_exposure+".fits"))
        print("Keeping galaxy model {}".format("test"))
        to_delete_list.remove(lightcurve.smphot_path.joinpath("galaxy_sn.fits"))

    for to_delete in to_delete_list:
        to_delete.unlink()

    return (returncode == 0)


def smphot_plot(lightcurve, logger, args):
    import matplotlib
    import matplotlib.pyplot as plt

    from utils import ListTable
    matplotlib.use('Agg')

    logger.info("Running smphot plots")
    sn_flux_df = ListTable.from_filename(lightcurve.smphot_path.joinpath("lightcurve_sn.dat")).df
    plot_path = lightcurve.smphot_path.joinpath("{}-{}_smphot_lightcurve.png".format(lightcurve.name, lightcurve.filterid))

    plt.subplots(figsize=(8., 3.))
    plt.errorbar(sn_flux_df['mjd'].to_numpy(), sn_flux_df['flux'].to_numpy(), yerr=sn_flux_df['varflux'].to_numpy(), fmt='.k')
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title("Calibrated lightcurve - {}-{}".format(lightcurve.name, lightcurve.filterid))
    plt.grid()
    plt.savefig(plot_path, dpi=300)

    plt.tight_layout()
    plt.close()

    logger.info("Plot can be found at: {}".format(plot_path))
    return True


def _run_star_mklc(i, smphot_stars_folder, mapping_folder, driver_path):
    #from deppol_utils import run_and_log
    stars_calib_cat_path = smphot_stars_folder.joinpath("mklc_{i}/calib_stars_cat_{i}.list".format(i=i))
    smphot_stars_cat_path = smphot_stars_folder.joinpath("mklc_{i}/smphot_stars_cat_{i}.list".format(i=i))
    _, return_log = run_and_log(["mklc", "-t", mapping_folder, "-O", smphot_stars_folder.joinpath("mklc_{}".format(i)), "-v", driver_path, "-o", smphot_stars_cat_path, '-c', stars_calib_cat_path, "-f", "1"], return_log=True)

    with open(smphot_stars_folder.joinpath("mklc_log_{}.log".format(i)), 'w') as f:
        f.write(return_log)


def smphot_stars(lightcurve, logger, args):
    import itertools
    import pandas as pd
    from dask import delayed, compute
    import numpy as np
    from utils import ListTable, contained_in_exposure, write_ds9_reg_circles
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.sparse import dok_matrix
    from itertools import chain

    lightcurve.smphot_stars_path.mkdir(exist_ok=True)

    ref_exposure = lightcurve.exposures[lightcurve.get_reference_exposure()]

    # First list all Gaia stars for wich there are measurements
    logger.info("Retrieving all Gaia stars for which there are measurements")
    cat_stars_df = lightcurve.get_ext_catalog('gaia', matched=True)
    cat_indices = []
    for exposure in lightcurve.get_exposures(files_to_check="cat_indices.hd5"):
        ext_cat_inside = exposure.get_catalog("cat_indices.hd5", key='ext_cat_inside')
        ext_cat_indices = exposure.get_catalog("cat_indices.hd5", key='ext_cat_indices')['indices']
        cat_indices.extend(np.arange(len(ext_cat_inside))[ext_cat_inside][ext_cat_indices].tolist())

    cat_indices = list(set(cat_indices))
    cat_stars_df = cat_stars_df.iloc[cat_indices]
    logger.info("Found {} stars".format(len(cat_stars_df)))

    # Remove stars that are outside of the reference quadrant
    logger.info("Removing stars outside of the reference quadrant")
    gaia_stars_skycoords = SkyCoord(ra=cat_stars_df['ra'], dec=cat_stars_df['dec'], unit='deg')
    inside = ref_exposure.wcs.footprint_contains(gaia_stars_skycoords)
    gaia_stars_skycoords = gaia_stars_skycoords[inside]
    cat_stars_df = cat_stars_df.loc[inside]
    logger.info("{} stars remaining".format(len(cat_stars_df)))

    # Remove stars that are too close of each other
    logger.info("Removing stars that are too close (20 as)")
    min_dist = 40/3600
    n = len(cat_stars_df)
    X = np.tile(cat_stars_df['ra'].to_numpy(), (n, 1))
    Y = np.tile(cat_stars_df['dec'].to_numpy(), (n, 1))
    dist = np.sqrt((X-X.T)**2+(Y-Y.T)**2)
    dist_mask = (dist <= min_dist)
    sp = dok_matrix(dist_mask)
    keys = list(filter(lambda x: x[0]!=x[1], list(sp.keys())))
    too_close_idx = list(set(list(chain(*keys))))
    keep_idx = list(filter(lambda x: x not in too_close_idx, range(n)))
    cat_stars_df = cat_stars_df.iloc[keep_idx]
    logger.info("{} stars remaining".format(len(cat_stars_df)))

    # Build dummy catalog with remaining Gaia stars
    logger.info("Building generic (empty) calibration catalog from Gaia stars")
    calib_df = pd.concat([cat_stars_df[['ra', 'dec']].reset_index(drop=True),
                          pd.DataFrame.from_dict({'n': np.ones(len(cat_stars_df), dtype=int),
                                                  'magg': np.zeros(len(cat_stars_df)),
                                                  'emagg': np.zeros(len(cat_stars_df)),
                                                  'magr': np.zeros(len(cat_stars_df)),
                                                  'emagr': np.zeros(len(cat_stars_df)),
                                                  'magi': np.zeros(len(cat_stars_df)),
                                                  'emagi': np.zeros(len(cat_stars_df))})], axis='columns').rename(columns={'Source': 'gaiaid'})

    # Output gaiaid list of calibration stars
    with open(lightcurve.smphot_stars_path.joinpath("stars_gaiaid.txt"), 'w') as f:
        for gaiaid in cat_stars_df['Source']:
            f.write("{}\n".format(gaiaid))

    logger.info("Total star count={}".format(len(calib_df)))

    # Do we still need this ?

    # logger.info("Removing faint stars (mag>=20)")
    # calib_df = calib_df.loc[calib_df['magg']<=20.]
    # logger.info("Selecting brigthest stars (up to mag=17)")
    # bright_calib_df = calib_df.loc[calib_df['magg']<=17.]
    # logger.info(" {} bright stars".format(len(bright_calib_df)))
    # calib_df = calib_df.loc[calib_df['magg']>17.]

    # disc_radius = 0.5*u.deg
    # sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')
    # sn_skycoord = SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg')
    # gaia_stars_skycoords = SkyCoord(ra=calib_df['ra'], dec=calib_df['dec'], unit='deg')

    # idxc, idxcatalog, d2d, d3d = sn_skycoord.search_around_sky(gaia_stars_skycoords, disc_radius)

    # inside_calib_df = calib_df.iloc[idxc]

    # logger.info("Total star count in a {} radius around SN: {} (excluding bright stars)".format(disc_radius, len(inside_calib_df)))
    # calib_df = pd.concat([bright_calib_df, inside_calib_df])
    # logger.info("Total stars: {}".format(len(calib_df)))

    write_ds9_reg_circles(lightcurve.path.joinpath("{}/smphot_stars_selection.reg".format(ref_exposure.name)), calib_df[['ra', 'dec']].to_numpy(), [20]*len(calib_df))
    lightcurve.smphot_stars_path.mkdir(exist_ok=True)

    calib_df['ristar'] = list(range(len(calib_df)))

    calib_table = ListTable(None, calib_df)

    stars_calib_cat_path = lightcurve.smphot_stars_path.joinpath("calib_stars_cat.list")
    smphot_stars_cat_path = lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list")
    calib_table.write_to(stars_calib_cat_path)

    if args.parallel_reduce:
        # If parallel reduce enable, split up the catalog
        logger.info("Running splitted mklc using {} workers".format(args.n_jobs))
        n = int(len(calib_df)/10)
        logger.info("Splitting into {}".format(n))
        calib_dfs = np.array_split(calib_df, n)
        jobs = []
        for i, calib_df in enumerate(calib_dfs):
            calib_stars_folder = lightcurve.smphot_stars_path.joinpath("mklc_{}".format(i))
            calib_stars_folder.mkdir(exist_ok=True)
            calib_table = ListTable(None, calib_df)
            calib_table.write_to(calib_stars_folder.joinpath("calib_stars_cat_{}.list".format(i)))

        logger.info("Submitting into scheduler")
        jobs = [delayed(_run_star_mklc)(i, lightcurve.smphot_stars_path, lightcurve.mappings_path, lightcurve.path.joinpath("smphot_driver")) for i in list(range(n))]
        compute(jobs)
        logger.info("Computation done, concatening catalogs")

        # Concatenate output catalog together
        calib_cat_paths = list(lightcurve.smphot_stars_path.glob("mklc_*/smphot_stars_cat_*.list"))
        calib_cat_tables = [ListTable.from_filename(calib_cat_path, delim_whitespace=False) for calib_cat_path in calib_cat_paths]
        [calib_cat_table.write_csv() for calib_cat_table in calib_cat_tables]

        calib_cat_df = pd.concat([calib_cat_table.df for calib_cat_table in calib_cat_tables])
        calib_cat_len = sum([int(calib_cat_table.header['nstars']) for calib_cat_table in calib_cat_tables])
        calib_cat_header = {'calibcatalog': stars_calib_cat_path, 'nstars': calib_cat_len, 'nimages': 0}

        calib_cat_table = ListTable(calib_cat_header, calib_cat_df)
        calib_cat_table.write_to(smphot_stars_cat_path)
        print(smphot_stars_cat_path.with_suffix(".csv"))
        calib_cat_table.write_to_csv(smphot_stars_cat_path.with_suffix(".csv"))
        logger.info("Done")

        logger.info("Deleting unuseful *.fits files...")
        to_delete_list = list(lightcurve.smphot_stars_path.glob("mklc_*/*.fits"))
        for to_delete in to_delete_list:
            to_delete.unlink()

    else:
        # Run on a single worker
        logger.info("Running mklc onto the main worker")
        run_and_log(["mklc", "-t", lightcurve.mappings_path, "-O", lightcurve.smphot_stars_path, "-v", lightcurve.path.joinpath("smphot_driver"), "-o", smphot_stars_cat_path, '-c', stars_calib_cat_path, "-f", "1"], logger=logger)
        logger.info("Done")
        logger.info("Deleting unuseful *.fits files...")
        to_delete_list = list(lightcurve.smphot_stars_path.glob("*.fits"))
        for to_delete in to_delete_list:
            to_delete.unlink()

    return True


def smphot_stars_constant(lightcurve, logger, args):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from deppol_utils import update_yaml
    from utils import ListTable
    from croaks.match import NearestNeighAssoc
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver

    # Load SMP star lightcurves
    logger.info("Loading stars SMP lightcurve...")
    smphot_lc_table = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list"), delim_whitespace=False)
    logger.info("Found {} measurements".format(len(smphot_lc_table.df)))

    # Remove negative fluxes
    smphot_lc_df = smphot_lc_table.df.loc[smphot_lc_table.df['flux']>0.]
    logger.info("Removing negative fluxes, down to {} measurements".format(len(smphot_lc_df)))

    # Create dataproxy for the fit
    piedestal = 0.
    dp = DataProxy(smphot_lc_df[['flux', 'error', 'star', 'mjd']].to_records(), flux='flux', error='error', star='star', mjd='mjd')
    dp.make_index('star')
    dp.make_index('mjd')
    w = 1./np.sqrt(dp.error**2+piedestal**2)

    # Retrieve matching Gaia catalog to taf fitted constant stars
    gaia_df = lightcurve.get_ext_catalog('gaia').drop_duplicates(subset='Source').set_index('Source', drop=True)
    with open(lightcurve.smphot_stars_path.joinpath("stars_gaiaid.txt"), 'r') as f:
        gaiaids = list(map(lambda x: int(x.strip()), f.readlines()))

    gaia_df = gaia_df.loc[gaiaids]

    # Fit of the constant star model
    model = LinearModel(list(range(len(dp.nt))), dp.star_index, np.ones_like(dp.star, dtype=float))
    solver = RobustLinearSolver(model, dp.flux, weights=w)
    solver.model.params.free = solver.robust_solution()

    # Add fit imformation to the lightcurve dataframe
    smphot_lc_df = smphot_lc_df.assign(mean_flux=solver.model.params.free[dp.star_index],
                                       emean_flux=np.sqrt(solver.get_cov().diagonal())[dp.star_index],
                                       res=solver.get_res(dp.flux),
                                       bads=solver.bads)
    smphot_lc_df = smphot_lc_df.assign(mean_mag=-2.5*np.log10(smphot_lc_df['mean_flux']),
                                       emean_mag=2.5/np.log(10)*smphot_lc_df['emean_flux']/smphot_lc_df['mean_flux'])
    smphot_lc_df = smphot_lc_df.assign(mag=-2.5*np.log10(smphot_lc_df['flux']),
                                       emag=2.5/np.log(10)*smphot_lc_df['error']/smphot_lc_df['flux'])

    # smphot_lc_df = smphot_lc_df.assign(wres=smphot_lc_df['res']/smphot_lc_df['error'])

    # # Constant stars dataframe creation
    # stars_gaiaids = gaia_df.iloc[list(dp.star_map.keys())]
    # stars_df = pd.DataFrame(data={'mag': -2.5*np.log10(solver.model.params.free),
    #                               'emag': np.abs(2.5/np.log(10)*np.sqrt(solver.get_cov().diagonal())/solver.model.params.free),
    #                               'rms_mag': [smphot_lc_df.loc[~solver.bads].loc[smphot_lc_df.loc[~solver.bads]['star'] == star]['res'].std() for star in list(dp.star_map.keys())],
    #                               'chi2': np.bincount(dp.star_index[~solver.bads], weights=smphot_lc_df.loc[~solver.bads]['wres']**2)/np.bincount(dp.star_index[~solver.bads]),
    #                               'gaiaid': gaia_df.iloc[list(dp.star_map.keys())].index.tolist()})

    # stars_df.set_index('gaiaid', drop=True, inplace=True)

    # # Create dataproxy for the fit
    # piedestal = 0.

    # smphot_lc_df = smphot_lc_df.assign(mag=-2.5*np.log10(smphot_lc_df['flux']),
    #                                    emag=np.abs(2.5/np.log(10)*smphot_lc_df['error']/smphot_lc_df['flux']))
    # dp = DataProxy(smphot_lc_df[['mag', 'emag', 'star', 'mjd']].to_records(), mag='mag', emag='emag', star='star', mjd='mjd')
    # dp.make_index('star')
    # dp.make_index('mjd')
    # w = 1./np.sqrt(dp.emag**2+piedestal**2)

    # # Retrieve matching Gaia catalog to taf fitted constant stars
    # gaia_df = lightcurve.get_ext_catalog('gaia').set_index('Source', drop=True)
    # with open(lightcurve.smphot_stars_path.joinpath("stars_gaiaid.txt"), 'r') as f:
    #     gaiaids = list(map(lambda x: int(x.strip()), f.readlines()))

    # gaia_df = gaia_df.loc[gaiaids]

    # # Fit of the constant star model
    # model = LinearModel(list(range(len(dp.nt))), dp.star_index, np.ones_like(dp.star, dtype=float))
    # solver = RobustLinearSolver(model, dp.mag, weights=w)
    # solver.model.params.free = solver.robust_solution()

    # # Add fit imformation to the lightcurve dataframe
    # smphot_lc_df = smphot_lc_df.assign(mean_mag=solver.model.params.free[dp.star_index],
    #                                    emean_mag=np.sqrt(solver.get_cov().diagonal())[dp.star_index],
    #                                    res=solver.get_res(dp.mag),
    #                                    bads=solver.bads)

    smphot_lc_df = smphot_lc_df.assign(wres=smphot_lc_df['res']/smphot_lc_df['error'])
    # smphot_lc_df = smphot_lc_df.assign(wres=smphot_lc_df['res']/smphot_lc_df['emag'])

    # Constant stars dataframe creation
    stars_gaiaids = gaia_df.iloc[list(dp.star_map.keys())]
    stars_df = pd.DataFrame(data={'mag': -2.5*np.log10(solver.model.params.free),
                                  'emag': 2.5/np.log(10)*np.sqrt(solver.get_cov().diagonal())/solver.model.params.free,
                                  'rms_mag': [(-2.5*np.log10(smphot_lc_df.loc[~solver.bads].loc[smphot_lc_df.loc[~solver.bads]['star'] == star]['mean_flux'])+2.5*np.log10(smphot_lc_df.loc[~solver.bads].loc[smphot_lc_df.loc[~solver.bads]['star']==star]['flux'])).std() for star in list(dp.star_map.keys())],
                                  'chi2': np.bincount(dp.star_index[~solver.bads], weights=smphot_lc_df.loc[~solver.bads]['wres']**2)/np.bincount(dp.star_index[~solver.bads]),
                                  'gaiaid': gaia_df.iloc[list(dp.star_map.keys())].index.tolist(),
                                  'star': dp.star_map.keys()})

    stars_df.set_index('gaiaid', inplace=True)

    # stars_gaiaids = gaia_df.iloc[list(dp.star_map.keys())]
    # stars_df = pd.DataFrame(data={'mag': solver.model.params.free,
    #                               'emag': np.sqrt(solver.get_cov().diagonal()),
    #                               # 'rms_mag': [smphot_lc_df.loc[~solver.bads].loc[smphot_lc_df.loc[~solver.bads]['star'] == star]['res'].std() for star in list(dp.star_map.keys())],
    #                               'chi2': np.bincount(dp.star_index[~solver.bads], weights=smphot_lc_df.loc[~solver.bads]['wres']**2)/np.bincount(dp.star_index[~solver.bads]),
    #                               'gaiaid': gaia_df.iloc[list(dp.star_map.keys())].index.tolist()})

    # Everything gets saved !
    smphot_lc_df.to_parquet(lightcurve.smphot_stars_path.joinpath("stars_lightcurves.parquet"))
    stars_df.to_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    logger.info("Done")

    # Update lightcurve yaml with fit informations
    logger.info("Updating lightcurve yaml")
    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'constant_stars',
                {'star_count': len(stars_df),
                 'chi2': np.sum(smphot_lc_df['wres']).item(),
                 'chi2/ndof': np.sum(smphot_lc_df['wres']).item()/len(stars_df),
                 'ndof': len(stars_df),
                 'piedestal': piedestal})

    return True


def smphot_stars_plot(lightcurve, logger, args):
    from utils import ListTable
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import numpy as np
    import pandas as pd
    import matplotlib
    from saunerie.plottools import binplot
    from scipy import stats
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver
    matplotlib.use('Agg')

    # Create plot output folders
    smphot_stars_plot_output = lightcurve.smphot_stars_path.joinpath("plots")
    smphot_stars_plot_output.mkdir(exist_ok=True)

    star_lc_folder = smphot_stars_plot_output.joinpath("lc_plots")
    star_lc_folder.mkdir(exist_ok=True)

    # Load catalogs
    stars_lc_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("stars_lightcurves.parquet"))
    stars_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    gaia_df = lightcurve.get_ext_catalog('gaia').reset_index().set_index('Source', drop=False).loc[stars_df.index]

    print(stars_df)

    stars_lc_outlier_df = stars_lc_df.loc[stars_lc_df['bads']]
    stars_lc_df = stars_lc_df.loc[~stars_lc_df['bads']]

    # Compare fitted star magnitude to catalog
    plt.subplots(figsize=(5., 5.))
    plt.suptitle("Fitted star magnitude vs external catalog ({})".format(args.photom_cat))
    plt.scatter(stars_df['mag'].to_numpy(), gaia_df['Gmag'].to_numpy(), c=gaia_df['BP-RP'].to_numpy(), s=10)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$m_\\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    cbar = plt.colorbar()
    cbar.set_label("$c_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.grid()
    plt.savefig(smphot_stars_plot_output.joinpath("mag_ext_cat.png"), dpi=300.)
    plt.close()

    # Repeatability plot
    plt.subplots(figsize=(8., 5.))
    plt.suptitle("Repeatability as a function of star magnitude")
    plt.plot(gaia_df['Gmag'].to_numpy(), stars_df['rms_mag'].to_numpy(), '.')
    plt.axhline(0.01, ls='-.', color='black', label="1%")
    plt.axhline(0.02, ls='--', color='black', label="2%")
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [AB mag]".format(args.photom_cat))
    plt.ylabel("$\sigma_\hat{m}$ [mag]")
    plt.legend()
    plt.savefig(smphot_stars_plot_output.joinpath("repeatability_mag.png"), dpi=300.)
    plt.close()

    # Same but zoomed in
    plt.subplots(figsize=(8., 5.))
    plt.suptitle("Repeatability as a function of star magnitude")
    plt.plot(gaia_df['Gmag'].to_numpy(), stars_df['rms_mag'].to_numpy(), '.')
    plt.axhline(0.01, ls='-.', color='black', label="1%")
    plt.axhline(0.02, ls='--', color='black', label="2%")
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [AB mag]".format(args.photom_cat))
    plt.ylabel("$\sigma_\hat{m}$ [mag]")
    plt.legend()
    plt.ylim(0., 0.05)
    plt.savefig(smphot_stars_plot_output.joinpath("repeatability_mag_zoomin.png"), dpi=300.)
    plt.close()

    # Star chi2
    plt.subplots(figsize=(8., 5.))
    plt.suptitle("$\chi^2$ / $m$")
    plt.plot(gaia_df['Gmag'].to_numpy(), stars_df['chi2'].to_numpy(), '.')
    plt.grid()
    plt.xlabel("$m$ [AB mag]")
    plt.ylabel("$\\chi^2$")
    plt.savefig(smphot_stars_plot_output.joinpath("mag_chi2.png"), dpi=300.)
    plt.close()


    res_min, res_max = -1000., 1000.
    x = np.linspace(res_min, res_max, 1000)
    m, s = norm.fit(stars_lc_df['res'])

    # Residual distribution
    plt.figure(figsize=(5., 5.))
    ax = plt.gca()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xlim(res_min, res_max)
    plt.plot(x, norm.pdf(x, loc=m, scale=s), color='black')
    plt.hist(stars_lc_df['res'].to_numpy(), bins=50, range=[res_min, res_max], density=True, histtype='step', color='black')
    plt.xlabel("$f_\mathrm{ADU}-\\left<f_\mathrm{ADU}\\right>$ [ADU]")
    plt.ylabel("density")
    plt.grid()
    plt.savefig(smphot_stars_plot_output.joinpath("residual_dist.png"), dpi=300.)
    plt.close()


    # plt.figure(figsize=(8., 4.))
    # plt.suptitle("Measure incertitude vs sky level")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['emean_flux'].to_numpy(), stars_lc_df['sky'].to_numpy(), ',', color='black')
    # # plt.xlim(0., 0.3)
    # # plt.ylim(-100., 100.)
    # plt.xlabel("$\\sigma_m$ [mag]")
    # plt.ylabel("sky [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_plot_output.joinpath("sky_var.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(8., 4.))
    # plt.suptitle("PSF measured star magnitude vs sky level")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['m'].to_numpy(), stars_lc_df['sky'].to_numpy(), ',', color='black')
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("sky [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("mag_sky.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(5., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['m'].to_numpy(), stars_lc_df['mag'].to_numpy(), ',', color='black')
    # plt.title("sky")
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$m$ [PS1 mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("mag_mag_ps1.png"), dpi=300.)
    # plt.close()


    # Fitted magnitude error vs. AB mag
    plt.figure(figsize=(8., 4.))
    ax = plt.gca()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(gaia_df['Gmag'].to_numpy(), stars_df['emag'].to_numpy(), '.', color='black')
    plt.ylim(0., 0.03)
    plt.xlabel("$m$ [AB mag]")
    plt.ylabel("$\\sigma_\hat{m}$ [mag]")
    plt.grid()
    plt.savefig(smphot_stars_plot_output.joinpath("mag_var.png"), dpi=300.)
    plt.close()

    # plt.figure()
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['em'].to_numpy(), stars_lc_df['res'].to_numpy(), ',', color='black')
    # plt.grid()
    # plt.xlim(0., 0.3)
    # plt.xlabel("$\sigma_m$ [mag]")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.savefig(smphot_stars_output.joinpath("var_residuals.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(10., 4.))
    # plt.title("Residuals as a function of airmass")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['airmass'].to_numpy(), stars_lc_df['res'].to_numpy(), ',')
    # plt.ylim(-0.5, 0.5)
    # plt.xlabel("$X$")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("airmass_mag.png"), dpi=300.)
    # plt.close()


    # plt.subplots(nrows=2, ncols=1, figsize=(10., 5.), gridspec_kw={'hspace': 0.}, sharex=True)
    # plt.suptitle("Standardized residuals")
    # ax = plt.subplot(2, 1, 1)
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # #xbinned_mag, yplot_student_res, res_dispersion = binplot(stars_lc_df['mag'].to_numpy(), (stars_lc_df['res']/stars_lc_df['em']).to_numpy(), nbins=10, data=True, rms=False, scale=False)
    # xbinned_mag, yplot_student_res, res_dispersion = binplot(stars_lc_df['mag'].to_numpy(), w*stars_lc_df['res'].to_numpy(), nbins=10, data=True, rms=False, scale=False)
    # plt.ylabel("$\\frac{m-\\left<m\\right>}{\\sigma_m}$")
    # plt.grid()

    # ax = plt.subplot(2, 1, 2)
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(xbinned_mag, res_dispersion, color='black')
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$\\frac{m-\\left<m\\right>}{\\sigma_m}$")

    # plt.savefig(smphot_stars_output.joinpath("student.png"), dpi=500.)
    # plt.close()

    # plt.subplots(nrows=2, ncols=1, figsize=(10., 5.), gridspec_kw={'hspace': 0.}, sharex=True)
    # plt.suptitle("Binned residuals")
    # ax = plt.subplot(2, 1, 1)
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid()
    # plt.ylim(-1., 1.)
    # xbinned_mag, yplot_student_res, res_dispersion = binplot(stars_lc_df['mag'].to_numpy(), stars_lc_df['res'].to_numpy(), nbins=10, data=True, rms=False, scale=False)
    # plt.ylabel("$m-\\left<m\\right>} [mag]$")

    # ax = plt.subplot(2, 1, 2)
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(xbinned_mag, res_dispersion, color='black')
    # plt.grid()
    # plt.xlabel("$m$ [AB mag]")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")

    # plt.savefig(smphot_stars_output.joinpath("binned_res.png"), dpi=500.)
    # plt.close()
    mjd_min, mjd_max = stars_lc_df['mjd'].min(), stars_lc_df['mjd'].max()

    # sigmas = [stars_lc_df.loc[stars_lc_df['star']==star_index]['res'].std() for star_index in list(set(stars_lc_df['star']))]
    # print(sum(np.array(sigmas)<=0.01))
    # print(sum(np.array(sigmas)<=0.02))
    # print(sum(np.array(sigmas)<=0.05))
    # plt.xlim(0., 0.2)
    # plt.grid()
    # plt.axvline(0.01, color='black')
    # plt.axvline(0.02, color='black')
    # plt.axvline(0.05, color='black')
    # plt.hist(sigmas, bins=50, histtype='step', range=[0., 0.2])
    # plt.show()
    stars_df = stars_df.reset_index().set_index('star')
    print(stars_df)
    for star_index in stars_df.index:
        star_mask = (stars_lc_df['star'] == star_index)
        outlier_star_mask = (stars_lc_outlier_df['star'] == star_index)
        if sum(star_mask) == 0 or np.any(stars_lc_df.loc[star_mask]['flux'] <= 0.):
            continue

        m = stars_df.loc[star_index]['mag']
        em = stars_df.loc[star_index]['rms_mag']
        cat_mag = gaia_df.loc[stars_df.loc[star_index]['gaiaid']]['Gmag']
        print("Star n°{}, G={}, m={}, sigma_m={}".format(star_index, cat_mag, m, em))

        fig = plt.subplots(ncols=2, nrows=1, figsize=(12., 4.), gridspec_kw={'width_ratios': [5, 1], 'wspace': 0, 'hspace': 0}, sharey=True)
        plt.suptitle("Star {} - $m_\mathrm{{{}}}$={} [AB mag]\n$m={:.4f}$ [mag], $\\sigma_m={:.4f}$ [mag]".format(star_index, args.photom_cat, cat_mag, m, em))
        ax = plt.subplot(1, 2, 1)
        plt.xlim(mjd_min, mjd_max)
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.errorbar(stars_lc_df.loc[star_mask]['mjd'].to_numpy(), stars_lc_df.loc[star_mask]['mag'].to_numpy(), yerr=stars_lc_df.loc[star_mask]['emag'].to_numpy(), marker='.', color='black', ls='')
        if len(outlier_star_mask) > 0.:
            plt.errorbar(stars_lc_outlier_df.loc[outlier_star_mask]['mjd'].to_numpy(), stars_lc_outlier_df.loc[outlier_star_mask]['mag'].to_numpy(), yerr=stars_lc_outlier_df.loc[outlier_star_mask]['emag'].to_numpy(), marker='x', color='black', ls='')
        plt.axhline(m, color='black')
        plt.grid(linestyle='--', color='black')
        plt.xlabel("MJD")
        plt.ylabel("$m$")
        plt.fill_between([stars_lc_df.loc[star_mask]['mjd'].min(), stars_lc_df.loc[star_mask]['mjd'].max()], [m+2.*em, m+2.*em], [m-2.*em, m-2.*em], color='xkcd:light blue')
        plt.fill_between([stars_lc_df.loc[star_mask]['mjd'].min(), stars_lc_df.loc[star_mask]['mjd'].max()], [m+em, m+em], [m-em, m-em], color='xkcd:sky blue')
        ax = plt.subplot(1, 2, 2)
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xticklabels([])
        x = np.linspace(stars_lc_df.loc[star_mask]['mag'].min(), stars_lc_df.loc[star_mask]['mag'].max(), 100)
        plt.plot(norm.pdf(x, loc=m, scale=em), x)
        plt.hist(stars_lc_df.loc[star_mask]['mag'], orientation='horizontal', density=True, bins='auto', histtype='step')

        plt.savefig(star_lc_folder.joinpath("star_{}.png".format(star_index)), dpi=250.)
        plt.close()

    return True


def smphot_flux_bias(lightcurve, logger, args):
    import pickle
    import numpy as np
    import pandas as pd
    from utils import ListTable, match_pixel_space
    from croaks.match import NearestNeighAssoc
    import matplotlib.pyplot as plt
    import copy

    exposures = lightcurve.get_exposures()
    smp_stars_lc_df = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list")).df
    with open(lightcurve.astrometry_path.joinpath("models.pickle"), 'rb') as f:
        astro_models = pickle.load(f)
        tp2px_model = astro_models['tp2px']
        astro_dp = astro_models['dp']


    smp_stars_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet")).dropna(subset=['ra', 'dec'])
    plt.plot(smp_stars_df['m'].to_numpy(), smp_stars_df['cat_mag'], '.')
    plt.show()
    tp2px_residuals = tp2px_model.residuals(np.array([astro_dp.tpx, astro_dp.tpy]), np.array([astro_dp.x, astro_dp.y]), np.array([astro_dp.pmtpx, astro_dp.pmtpy]), astro_dp.mjd, exposure_indices=astro_dp.exposure_index)
    astro_stars = {}
    for gaiaid in astro_dp.gaiaid_map.keys():
        star_mask = (astro_dp.gaiaid == gaiaid)
        sigma_x = np.std(tp2px_residuals[0][star_mask])
        sigma_y = np.std(tp2px_residuals[1][star_mask])
        cat_mag = astro_dp.cat_mag[star_mask][0]
        astro_stars[gaiaid] = {'sx': sigma_x, 'sy': sigma_y, 'ra': np.mean(astro_dp.ra[star_mask]), 'dec': np.mean(astro_dp.dec[star_mask]), 'cat_mag': cat_mag, 'gaiaid': gaiaid}

    astro_stars_df = pd.DataFrame.from_dict(data=astro_stars, orient='index')

    assoc = NearestNeighAssoc(first=[astro_stars_df['ra'].to_numpy(), astro_stars_df['dec'].to_numpy()], radius=1./60./60.)
    i = assoc.match(smp_stars_df['ra'].to_numpy(), smp_stars_df['dec'].to_numpy())

    astro_stars_df = astro_stars_df.iloc[i[i>=0]].reset_index(drop=True)
    smp_stars_df = smp_stars_df.iloc[i>=0].reset_index(drop=True)

    smp_stars_df['gaiaid'] = astro_stars_df['gaiaid']

    import pathlib

    if not pathlib.Path("deltas.pickle").exists():
        delta_mags = []
        delta_astros = []
        for exposure in exposures:
            psf_cat_df = exposure.get_matched_catalog('psfstars')
            gaia_cat_df = exposure.get_matched_ext_catalog('gaia')
            exp_smp_stars_df = smp_stars_lc_df.loc[smp_stars_lc_df['name']==exposure.name]
            exp_astro_dp = copy.deepcopy(astro_dp)
            exp_astro_dp.compress(astro_dp.exposure==exposure.name)
            exp_astro_residuals = tp2px_residuals[:, astro_dp.exposure==exposure.name]

            assoc = NearestNeighAssoc(first=[gaia_cat_df['ra'].to_numpy(), gaia_cat_df['dec'].to_numpy()], radius=1./60./60.)
            i = assoc.match(exp_smp_stars_df['ra'].to_numpy(), exp_smp_stars_df['dec'].to_numpy())

            psf_cat_df = psf_cat_df.iloc[i[i>=0]].reset_index(drop=True)
            gaia_cat_df = gaia_cat_df.iloc[i[i>=0]].reset_index(drop=True)
            exp_smp_stars_df = exp_smp_stars_df.iloc[i>=0].reset_index(drop=True)

            for gaiaid in gaia_cat_df.Source:
                delta_astro = exp_astro_residuals[:, exp_astro_dp.gaiaid==gaiaid].reshape(2)
                delta_mag = -2.5*np.log10(exp_smp_stars_df.loc[gaia_cat_df['Source']==gaiaid]['flux'].to_numpy()) - smp_stars_df.loc[smp_stars_df['gaiaid']==gaiaid]['m'].to_numpy()

                if len(delta_mag) == 1:
                    delta_mags.append(delta_mag)
                    delta_astros.append(delta_astro)

        delta_astros = np.array(delta_astros)
        delta_mags = np.array(delta_mags)
        with open("deltas.pickle", 'wb') as f:
            pickle.dump([delta_astros, delta_mags], f)
    else:
        with open("deltas.pickle", 'rb') as f:
            delta_astros, delta_mags = pickle.load(f)

    plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(delta_mags, delta_astros[:, 0], ',')
    plt.xlabel("$\Delta m$ [mag]")
    plt.ylabel("$\Delta x$ [pixel]")

    plt.subplot(1, 2, 2)
    plt.plot(delta_mags, delta_astros[:, 1], ',')
    plt.xlabel("$\Delta m$ [mag]")
    plt.ylabel("$\Delta x$ [pixel]")

    plt.show()
    plt.close()
    return True

    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.suptitle("Astrometry repeatability as a function of star magnitude (Gaia $G$ band)")
    plt.subplot(1, 2, 1)
    plt.plot(astro_stars_df['cat_mag'].to_numpy(), astro_stars_df['sx'], ',')
    plt.ylim(0., 0.2)
    plt.xlabel("G [mag]")
    plt.ylabel("$\sqrt{x-x_\mathrm{model}}$")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(astro_stars_df['cat_mag'].to_numpy(), astro_stars_df['sy'], ',')
    plt.ylim(0., 0.2)
    plt.xlabel("G [mag]")
    plt.ylabel("$\sqrt{y-y_\mathrm{model}}$")
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.smphot_stars_path.joinpath("astro_repeatability.png"), dpi=150.)
    plt.close()

    plt.subplots(nrows=1, ncols=1, figsize=(15., 7.))
    plt.subplot(1, 2, 1)
    # plt.plot(astro_stars_df.sample(frac=1)['sx'].to_numpy(), smp_stars_df.sample(frac=1)['sigma_m'].to_numpy(),  '.')
    plt.plot(astro_stars_df['sx'].to_numpy(), smp_stars_df['sigma_m'].to_numpy(),  ',')
    plt.xlabel("Astrometry precision - $x$ axis [pixel]")
    plt.ylabel("Photometry repeatability [mag]")
    plt.grid()

    plt.subplot(1, 2, 2)
    # plt.plot(astro_stars_df.sample(frac=1)['sy'].to_numpy(), smp_stars_df.sample(frac=1)['sigma_m'].to_numpy(),  '.') #
    plt.plot(astro_stars_df['sy'].to_numpy(), smp_stars_df['sigma_m'].to_numpy(),  ',')
    plt.xlabel("Astrometry precision - $y$ axis [pixel]")
    plt.ylabel("Photometry repeatability [mag]")
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.smphot_stars_path.joinpath("astro_smp_flux_bias.png"), dpi=150.)
    plt.close()
