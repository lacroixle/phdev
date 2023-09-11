#!/usr/bin/env python3

from deppol_utils import run_and_log


def reference_quadrant(lightcurve, logger, args):
    """

    """
    # TODO: Rewrite this using quadrants.parquet instead of reading all fits headers
    import pathlib
    import pandas as pd
    import numpy as np

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    exposures = lightcurve.get_exposures(files_to_check="psfstars.list")
    seeings = dict([(exposure.name, (exposure.exposure_header['seeing'], exposure.field)) for exposure in exposures])
    seeings_df = pd.DataFrame({'seeing': list(map(lambda x: x.exposure_header['seeing'], exposures)),
                               'fieldid': list(map(lambda x: x.field, exposures)),
                               'star_count': list(map(lambda x: len(x.get_matched_catalog('psfstars')), exposures))},
                              index=list(map(lambda x: x.name, exposures)))

    print(seeings_df.sort_values('star_count'))

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

    plt.errorbar(sn_flux_df['mjd'].to_numpy(), sn_flux_df['flux'].to_numpy(), yerr=sn_flux_df['varflux'].to_numpy(), fmt='.k')
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title("Calibrated lightcurve - {}-{}".format(lightcurve.name, lightcurve.filterid))
    plt.grid()
    plt.savefig(plot_path, dpi=300)
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
    from utils import ListTable, contained_in_exposure
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ref_exposure = lightcurve.exposures[lightcurve.get_reference_exposure()]

    # Build calibration catalog
    cat_stars_df = lightcurve.get_ext_catalog(args.photom_cat)

    import matplotlib.pyplot as plt

    # First list all calibration stars in the catalog
    cat_indices = []
    for exposure in lightcurve.get_exposures(files_to_check="cat_indices.hd5"):
        ext_cat_inside = exposure.get_catalog("cat_indices.hd5", key='ext_cat_inside')
        ext_cat_indices = exposure.get_catalog("cat_indices.hd5", key='ext_cat_indices')['indices']
        cat_indices.extend(np.arange(len(ext_cat_inside))[ext_cat_inside][ext_cat_indices].tolist())

    cat_indices = list(set(cat_indices))
    cat_stars_df = cat_stars_df.iloc[cat_indices]

    logger.info("Building calibration catalog using {}".format(args.photom_cat))
    if args.photom_cat == 'gaia':
        calib_df = cat_stars_df[['ra', 'dec', 'Gmag', 'RPmag', 'e_RPmag', 'e_Gmag']].rename(columns={'RPmag': 'magr', 'e_RPmag': 'emagr', 'Gmag': 'magg', 'e_Gmag': 'emagg'})
        calib_df['magi'] = calib_df['magr']
        calib_df['emagi'] = calib_df['emagr']
        calib_df.insert(2, column='n', value=1)

    elif args.photom_cat == 'ps1':
        cat_stars_df.dropna(subset=['gmag', 'rmag', 'imag', 'e_gmag', 'e_rmag', 'e_imag'], inplace=True)
        calib_df = cat_stars_df[['ra', 'dec', 'gmag', 'rmag', 'imag', 'e_gmag', 'e_rmag', 'e_imag']].rename(
            columns={'gmag': 'magg', 'rmag': 'magr', 'imag': 'magi', 'e_gmag': 'emagg', 'e_rmag': 'emagr', 'e_imag': 'emagi'})
        calib_df.insert(2, column='n', value=1)

    elif args.photom_cat == 'ubercal':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Filter stars by SN proximity (in some radius defined by --smphot-stars-radius) and magnitude
    logger.info("Total star count={}".format(len(calib_df)))

    logger.info("Removing faint stars (mag>=20)")
    calib_df = calib_df.loc[calib_df['magg']<=20.]
    logger.info("Selecting brigthest stars (up to mag=17)")
    bright_calib_df = calib_df.loc[calib_df['magg']<=17.]
    logger.info(" {} bright stars".format(len(bright_calib_df)))
    calib_df = calib_df.loc[calib_df['magg']>17.]

    disc_radius = 0.5*u.deg
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')
    sn_skycoord = SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg')
    gaia_stars_skycoords = SkyCoord(ra=calib_df['ra'], dec=calib_df['dec'], unit='deg')

    idxc, idxcatalog, d2d, d3d = sn_skycoord.search_around_sky(gaia_stars_skycoords, disc_radius)

    inside_calib_df = calib_df.iloc[idxc]

    logger.info("Total star count in a {} radius around SN: {} (excluding bright stars)".format(disc_radius, len(inside_calib_df)))
    calib_df = pd.concat([bright_calib_df, inside_calib_df])
    logger.info("Total stars: {}".format(len(calib_df)))

    logger.info("Removing stars outside of the reference quadrant")
    gaia_stars_skycoords = SkyCoord(ra=calib_df['ra'], dec=calib_df['dec'], unit='deg')
    inside = ref_exposure.wcs.footprint_contains(gaia_stars_skycoords)
    gaia_stars_skycoords = gaia_stars_skycoords[inside]
    calib_df = calib_df.loc[inside]
    logger.info("New star count={}".format(len(calib_df)))


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
        #calib_cat_paths = [smphot_stars_folder.joinpath("smphot_stars_cat_{}.list".format(i)) for i in range(n)]
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


class ConstantStarModel:
    def __init__(self):
        pass


def smphot_stars_constant(lightcurve, logger, args):
    from utils import ListTable
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver

    # Load SMP star lightcurves and calibration catalogs
    calib_table = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list"), delim_whitespace=False)
    cat_calib_table = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("calib_stars_cat.list"))

    stars_lc_df = calib_table.df.loc[calib_table.df['flux'] >= 0.]
    print("Removed {} negative fluxes".format(len(calib_table.df)-len(stars_lc_df)))

    # Create dataproxy
    piedestal = 0.015
    stars_lc_df['m'] = -2.5*np.log10(stars_lc_df['flux'])
    stars_lc_df['em'] = np.abs(-2.5/np.log(10)*stars_lc_df['error']/stars_lc_df['flux'])

    dp = DataProxy(stars_lc_df[['m', 'em', 'star', 'mjd']].to_records(), m='m', em='em', star='star', mjd='mjd')
    dp.make_index('m')
    dp.make_index('star')
    dp.make_index('mjd')
    w = 1./np.sqrt(dp.em**2+piedestal**2)

    # Fit of the constant star model
    model = LinearModel(list(range(len(dp.nt))), dp.star_index, np.ones_like(dp.m))
    solver = RobustLinearSolver(model, dp.m, weights=w)
    solver.model.params.free = solver.robust_solution()

    stars_lc_df['mean_m'] = solver.model.params.free[dp.star_index]
    stars_lc_df['res'] = solver.get_res(dp.m)
    stars_lc_df['wres'] = stars_lc_df['res']*w
    stars_lc_df['bad'] = solver.bads

    star_chi2 = np.bincount(dp.star_index[~solver.bads], weights=stars_lc_df.loc[~stars_lc_df['bad']]['wres']**2)/np.bincount(dp.star_index[~solver.bads])

    stars_df = pd.DataFrame(data={'m': solver.model.params.free, 'em': np.sqrt(solver.get_cov().diagonal()), 'chi2': star_chi2, 'star': dp.star_map.keys()})

    stars_df['cat_mag'] = cat_calib_table.df.iloc[stars_df['star']]['mag'+lightcurve.filterid[1]].tolist()
    stars_df['cat_emag'] = cat_calib_table.df.iloc[stars_df['star']]['emag'+lightcurve.filterid[1]].tolist()
    stars_df['cat_color'] = (cat_calib_table.df.iloc[stars_df['star']]['magi'] - cat_calib_table.df.iloc[stars_df['star']]['magg']).tolist()
    stars_df['sigma_m'] = [stars_lc_df.loc[~stars_lc_df['bad']].loc[stars_lc_df.loc[~stars_lc_df['bad']]['star'] == star]['res'].std() for star in stars_df['star'].tolist()]

    stars_df.set_index('star', drop=True, inplace=True)

    stars_lc_df.to_parquet(lightcurve.smphot_stars_path.joinpath("stars_lightcurves.parquet"))
    stars_df.to_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))


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
    # matplotlib.use('Agg')

    smphot_stars_plot_output = lightcurve.smphot_stars_path.joinpath("plots")
    smphot_stars_plot_output.mkdir(exist_ok=True)

    stars_lc_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("stars_lightcurves.parquet"))
    stars_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))

    star_lc_folder = lightcurve.smphot_stars_path.joinpath("lc_plots")
    star_lc_folder.mkdir(exist_ok=True)
    stars_lc_outlier_df = stars_lc_df.loc[stars_lc_df['bad']]
    stars_lc_df = stars_lc_df.loc[~stars_lc_df['bad']]


    plt.subplots(figsize=(5., 5.))
    plt.suptitle("Fitted star magnitude vs external catalog ({})".format(args.photom_cat))
    plt.scatter(stars_df['m'].to_numpy(), stars_df['cat_mag'].to_numpy(), c=stars_df['cat_color'].to_numpy(), s=10)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$m_\\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    cbar = plt.colorbar()
    cbar.set_label("$c_\mathrm{PS1}$ [mag]")
    plt.grid()
    plt.savefig(lightcurve.smphot_stars_path.joinpath("mag_ext_cat.png"), dpi=300.)
    plt.close()

    plt.subplots(figsize=(8., 5.))
    plt.suptitle("Repeatability as a function of star magnitude")
    plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['sigma_m'].to_numpy(), '.')
    plt.axhline(0.01, ls='-.', color='black', label="1%")
    plt.axhline(0.02, ls='--', color='black', label="2%")
    plt.grid()
    plt.xlabel("$m_\mathrm{PS1}$ [AB mag]")
    plt.ylabel("$\sigma_\hat{m}$ [mag]")
    plt.legend()
    plt.savefig(lightcurve.smphot_stars_path.joinpath("repeatability_mag.png"), dpi=300.)
    plt.close()

    plt.subplots(figsize=(8., 5.))
    plt.suptitle("Repeatability as a function of star magnitude")
    plt.plot(stars_df['cat_mag'].to_numpy(), stars_df['sigma_m'].to_numpy(), '.')
    plt.axhline(0.01, ls='-.', color='black', label="1%")
    plt.axhline(0.02, ls='--', color='black', label="2%")
    plt.grid()
    plt.xlabel("$m_\mathrm{PS1}$ [AB mag]")
    plt.ylabel("$\sigma_\hat{m}$ [mag]")
    plt.legend()
    plt.ylim(0., 0.05)
    plt.savefig(lightcurve.smphot_stars_path.joinpath("repeatability_mag_zoomin.png"), dpi=300.)
    plt.close()


    return

    # plt.hist(stars_df['m'], bins=15)
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("Count")
    # plt.show()

    # plt.plot(stars_df['m'].to_numpy(), stars_df['chi2'].to_numpy(), '.')
    # plt.grid()
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$\\chi^2$")
    # plt.show()

    # res_min, res_max = -0.5, 0.5
    # x = np.linspace(res_min, res_max, 1000)
    # m, s = norm.fit(stars_lc_df['res'])

    # plt.figure(figsize=(5., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.xlim(res_min, res_max)
    # plt.plot(x, norm.pdf(x, loc=m, scale=s), color='black')
    # plt.hist(stars_lc_df['res'].to_numpy(), bins=100, range=[res_min, res_max], density=True, histtype='step', color='black')
    # plt.xlabel("$m-\\left<m\\right>$ [mag]")
    # plt.ylabel("density")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("residual_dist.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(8., 4.))
    # plt.suptitle("Measure incertitude vs sky level")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['em'].to_numpy(), stars_lc_df['sky'].to_numpy(), ',', color='black')
    # plt.xlim(0., 0.3)
    # plt.ylim(-100., 100.)
    # plt.xlabel("$\\sigma_m$ [mag]")
    # plt.ylabel("sky [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("sky_var.png"), dpi=300.)
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


    # plt.figure(figsize=(8., 4.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['m'].to_numpy(), stars_lc_df['em'].to_numpy(), ',', color='black')
    # plt.ylim(0., 0.3)
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$\\sigma_m$")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("mag_var.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(8., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_lc_df['mag'].to_numpy(), stars_lc_df['res'].to_numpy(), ',', color='black')
    # plt.ylim(-1., 1.)
    # plt.xlabel("$m$ [PS1 mag]")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("residuals_mag.png"), dpi=300.)
    # plt.close()


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
    for star_index in list(set(stars_lc_df['star'])):
        star_mask = (stars_lc_df['star'] == star_index)
        outlier_star_mask = (stars_lc_outlier_df['star'] == star_index)
        if sum(star_mask) == 0 or np.any(stars_lc_df.loc[star_mask]['flux'] <= 0.):
            continue

        m = stars_df.loc[star_index]['m']
        em = stars_df.loc[star_index]['sigma_m']
        print("Star n°{}, m={}, sigma_m={}".format(star_index, m, em))

        fig = plt.subplots(ncols=2, nrows=1, figsize=(12., 4.), gridspec_kw={'width_ratios': [5, 1], 'wspace': 0, 'hspace': 0}, sharey=True)
        plt.suptitle("Star {} - $m_\mathrm{{PS1}}$={} [mag]\n$\\sigma={:.4f}$".format(star_index, stars_lc_df.loc[star_mask]['mag'].tolist()[0], em))
        ax = plt.subplot(1, 2, 1)
        plt.xlim(mjd_min, mjd_max)
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.errorbar(stars_lc_df.loc[star_mask]['mjd'].to_numpy(), stars_lc_df.loc[star_mask]['m'].to_numpy(), yerr=stars_lc_df.loc[star_mask]['em'].to_numpy(), marker='.', color='black', ls='')
        if len(outlier_star_mask) > 0.:
            plt.errorbar(stars_lc_outlier_df.loc[outlier_star_mask]['mjd'].to_numpy(), stars_lc_outlier_df.loc[outlier_star_mask]['m'].to_numpy(), yerr=stars_lc_outlier_df.loc[outlier_star_mask]['em'].to_numpy(), marker='x', color='black', ls='')
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
        x = np.linspace(stars_lc_df.loc[star_mask]['m'].min(), stars_lc_df.loc[star_mask]['m'].max(), 100)
        plt.plot(norm.pdf(x, loc=m, scale=em), x)
        plt.hist(stars_lc_df.loc[star_mask]['m'], orientation='horizontal', density=True, bins='auto', histtype='step')

        plt.savefig(star_lc_folder.joinpath("star_{}.png".format(star_index)), dpi=250.)
        plt.close()

    return True
