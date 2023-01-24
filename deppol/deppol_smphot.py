#!/usr/bin/env python3

from deppol_utils import run_and_log

def reference_quadrant(band_path, ztfname, filtercode, logger, args):
    """

    """
    # TODO: Rewrite this using quadrants.parquet instead of reading all fits headers
    import pathlib
    import pandas as pd
    import numpy as np
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from utils import get_header_from_quadrant_path, read_list
    from deppol_utils import quadrants_from_band_path

    # Determination of the best seeing quadrant
    # First determine the most represented field
    logger.info("Determining best seeing quadrant...")
    quadrant_paths = quadrants_from_band_path(band_path, logger, check_files="psfstars.list")

    seeings = {}
    for quadrant_path in quadrant_paths:
        quadrant_header = get_header_from_quadrant_path(quadrant_path)
        seeings[quadrant_path] = (quadrant_header['seseeing'], quadrant_header['fieldid'])

    fieldids = list(set([seeing[1] for seeing in seeings.values()]))
    fieldids_count = [sum([1 for f in seeings.values() if f[1]==fieldid]) for fieldid in fieldids]
    maxcount_field = fieldids[np.argmax(fieldids_count)]

    logger.info("Computing reference quadrants from {} quadrants".format(len(quadrant_paths)))
    logger.info("{} different field ids".format(len(fieldids)))
    logger.info("Field ids: {}".format(fieldids))
    logger.info("Count: {}".format(fieldids_count))
    logger.info("Max quadrant field={}".format(maxcount_field))

    seeing_df = pd.DataFrame([[quadrant, seeings[quadrant][0]] for quadrant in seeings.keys() if seeings[quadrant][1]==maxcount_field], columns=['quadrant', 'seeing'])
    seeing_df = seeing_df.set_index(['quadrant'])

    # Remove exposure where small amounts of stars are detected
    seeing_df['n_standalonestars'] = list(map(lambda x: len(read_list(pathlib.Path(x).joinpath("standalone_stars.list"))[1]), seeing_df.index))
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
    driver_path = band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))
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
        for quadrant_path in quadrant_paths:
            f.write("{}\n".format(quadrant_path))
        f.write("PHOREF\n")
        f.write("{}\n".format(idxmin))
        f.write("PMLIST\n")
        f.write(str(band_path.joinpath("astrometry/pmcatalog.list")))

    with open(band_path.joinpath("reference_quadrant"), 'w') as f:
        f.write(str(idxmin.name))


def smphot(band_path, ztfname, filtercode, logger, args):
    from deppol_utils import run_and_log

    logger.info("Running scene modeling")

    smphot_output = band_path.joinpath("smphot_output")
    smphot_output.mkdir(exist_ok=True)

    run_and_log(["mklc", "-t", band_path.joinpath("mappings"), "-O", smphot_output, "-v", band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))], logger=logger)

    return True


def smphot_plot(band_path, ztfname, filtercode, logger, args):
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # logger.info("Running pmfit plots")
    # driver_path = band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))
    # gaia_path = band_path.joinpath("{}/{}/gaia.npy".format(ztfname, filtercode))
    # run_and_log(["pmfit", driver_path, "--gaia={}".format(gaia_path), "--outdir={}".format(cwd.joinpath("pmfit")), "--plot-dir={}".format(cwd.joinpath("pmfit_plot")), "--plot", "--mu-max=20."], logger=logger)

    # logger.info("Running smphot plots")
    # with open(band_path.joinpath("smphot_output/lightcurve_sn.dat"), 'r') as f:
    #     _, sn_flux_df = list_format.read_list(f)

    # plt.errorbar(sn_flux_df['mjd'], sn_flux_df['flux'], yerr=sn_flux_df['varflux'], fmt='.k')
    # plt.xlabel("MJD")
    # plt.ylabel("Flux")
    # plt.title("Calibrated lightcurve - {} - {}".format(ztfname, filtercode))
    # plt.grid()
    # plt.savefig(band_path.joinpath("{}-{}_smphot_lightcurve.png".format(ztfname, filtercode)), dpi=300)
    # plt.close()

    return True


def _run_star_mklc(i, smphot_stars_folder, mapping_folder, driver_path):
    #from deppol_utils import run_and_log
    stars_calib_cat_path = smphot_stars_folder.joinpath("calib_stars_cat_{}.list".format(i))
    smphot_stars_cat_path = smphot_stars_folder.joinpath("smphot_stars_cat_{}.list".format(i))
    _, return_log = run_and_log(["mklc", "-t", mapping_folder, "-O", smphot_stars_folder, "-v", driver_path, "-o", smphot_stars_cat_path, '-c', stars_calib_cat_path, "-f", "1"], return_log=True)

    with open(smphot_stars_folder.joinpath("mklc_log_{}.log".format(i)), 'w') as f:
        f.write(return_log)


def smphot_stars(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    from dask import delayed, compute
    import numpy as np
    from utils import ListTable
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Build calibration catalog
    gaia_stars_df = pd.read_parquet(band_path.joinpath("gaia_stars.parquet"))

    calib_df = gaia_stars_df[['ra', 'dec', 'gmag', 'rpmag', 'e_rpmag', 'e_gmag']].rename(columns={'rpmag': 'magr', 'e_rpmag': 'emagr', 'gmag': 'magg', 'e_gmag': 'emagg'})
    calib_df.insert(2, column='n', value=1)
    calib_df['ristar'] = list(range(len(gaia_stars_df)))

    gaia_stars_skycoords = SkyCoord(ra=gaia_stars_df['ra'], dec=gaia_stars_df['dec'], unit='deg')

    # Filter stars by SN proximity (in some radius defined by --smphot-stars-radius)
    sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')
    sn_skycoord = SkyCoord(ra=sn_parameters['sn_ra'], dec=sn_parameters['sn_dec'], unit='deg')
    idxc, idxcatalog, d2d, d3d = sn_skycoord.search_around_sky(gaia_stars_skycoords, 0.35*u.deg)

    calib_df = calib_df.iloc[idxc]
    calib_table = ListTable(None, calib_df)

    smphot_stars_folder = band_path.joinpath("smphot_stars")
    smphot_stars_folder.mkdir(exist_ok=True)

    stars_calib_cat_path = smphot_stars_folder.joinpath("calib_stars_cat.list")
    calib_table.write_to(stars_calib_cat_path)

    driver_path = band_path.joinpath("{}_driver_{}".format(ztfname, filtercode))
    mapping_folder = band_path.joinpath("mappings")
    smphot_stars_cat_path = smphot_stars_folder.joinpath("smphot_stars_cat.list")

    if args.parallel_reduce:
        # If parallel reduce enable, split up the catalog
        n = int(len(calib_df)/4)
        calib_dfs = np.array_split(calib_df, n)
        jobs = []
        for i, calib_df in enumerate(calib_dfs):
            calib_table = ListTable(None, calib_df)
            calib_table.write_to(smphot_stars_folder.joinpath("calib_stars_cat_{}.list".format(i)))

        jobs = [delayed(_run_star_mklc)(i, smphot_stars_folder, mapping_folder, driver_path) for i in list(range(n))]
        compute(jobs)

        # Concatenate output catalog together
        calib_cat_paths = [smphot_stars_folder.joinpath("smphot_stars_cat_{}.list".format(i)) for i in range(n)]
        calib_cat_tables = [ListTable.from_filename(calib_cat_path) for calib_cat_path in calib_cat_paths]

        calib_cat_df = pd.concat([calib_cat_table.df for calib_cat_table in calib_cat_tables])
        calib_cat_len = sum([int(calib_cat_table.header['nstars']) for calib_cat_table in calib_cat_tables])
        calib_cat_header = {'calibcatalog': stars_calib_cat_path, 'nstars': calib_cat_len, 'nimages': 0}

        calib_cat_table = ListTable(calib_cat_header, calib_cat_df)
        calib_cat_table.write_to(smphot_stars_cat_path)

    else:
        # Run on a single worker
        run_and_log(["mklc", "-t", band_path.joinpath("mappings"), "-O", smphot_stars_folder, "-v", driver_path, "-o", smphot_stars_cat_path, '-c', stars_calib_cat_path, "-f", "1"], logger=logger)



class ConstantStarModel:
    def __init__(self):
        pass


def smphot_stars_plot(band_path, ztfname, filtercode, logger, args):
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

    smphot_stars_output = band_path.joinpath("smphot_stars_output")
    calib_table = ListTable.from_filename(smphot_stars_output.joinpath("calib_cat.list"))

    stars_df = calib_table.df[calib_table.df['flux'] >= 0.]
    # Remove negative fluxes
    stars_df = stars_df[stars_df['flux']>=0.]
    print("Removed {} negative fluxes".format(len(calib_table.df)-len(stars_df)))


    # Create dataproxy
    #piedestal = 0.0005
    piedestal = 0.00015
    stars_df['m'] = -2.5*np.log10(stars_df['flux'])
    #stars_df['em'] = np.abs(-2.5/np.log(10)*stars_df['error']/stars_df['flux'])
    stars_df['em'] = np.sqrt(np.abs(-2.5/np.log(10)*stars_df['error']/stars_df['flux'])**2+piedestal)

    dp = DataProxy(stars_df[['m', 'em', 'star']].to_records(), m='m', em='em', star='star')
    dp.make_index('star')
    w = 1./np.sqrt(dp.em**2+piedestal)

    model = LinearModel(list(range(len(dp.nt))), dp.star_index, np.ones_like(dp.m))
    solver = RobustLinearSolver(model, dp.m, weights=1./dp.em)
    solver.model.params.free = solver.robust_solution()

    # # Remove outliers
    # star_dfs = [stars_df[stars_df['star']==star] for star in list(set(stars_df['star'].tolist()))]

    # star_dfs = []
    # for star in set(stars_df['star'].tolist()):
    #     star_df = stars_df[stars_df['star']==star]
    #     to_keep = (np.abs(stats.zscore(star_df['m'])) < 4.)
    #     star_dfs.append(star_df[to_keep])

    # before_or = len(stars_df)
    # stars_df = pd.concat(star_dfs)
    # print("Removed {} outliers (out of {})".format(before_or-len(stars_df), before_or))

    star_lc_folder = smphot_stars_output.joinpath("lc_plots")
    star_lc_folder.mkdir(exist_ok=True)

    # mean_m = np.array([np.sum(stars_df.loc[stars_df['star']==star_index]['m']/stars_df.loc[stars_df['star']==star_index]['em']**2)/np.sum(1/stars_df.loc[stars_df['star']==star_index]['em']**2) for star_index in range(max(list(set(stars_df['star'])))+1)])
    # mean_m = np.nan_to_num(mean_m)
    # stars_df['mean_m'] = mean_m[stars_df['star'].tolist()]
    # stars_df['res'] = (stars_df['m'] - stars_df['mean_m']).dropna(axis='rows')
    stars_df['mean_m'] = solver.model.params.free[dp.star_index]
    stars_df['res'] = solver.get_res(dp.m)

    outlier_stars_df = stars_df[solver.bads]
    stars_df = stars_df[~solver.bads]
    w = w[~solver.bads]

    # res_min, res_max = stars_df['res'].min(), stars_df['res'].max()
    # res_min, res_max = -0.5, 0.5
    # x = np.linspace(res_min, res_max, 1000)
    # m, s = norm.fit(stars_df['res'])

    # plt.figure(figsize=(5., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.xlim(res_min, res_max)
    # plt.plot(x, norm.pdf(x, loc=m, scale=s), color='black')
    # plt.hist(stars_df['res'], bins=100, range=[res_min, res_max], density=True, histtype='step', color='black')
    # plt.xlabel("$m-\\left<m\\right>$ [mag]")
    # plt.ylabel("density")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("residual_dist.png"), dpi=300.)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(8., 4.))
    # plt.suptitle("Measure incertitude vs sky level")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['em'], stars_df['sky'], ',', color='black')
    # plt.xlim(0., 0.3)
    # plt.ylim(-100., 100.)
    # plt.xlabel("$\\sigma_m$ [mag]")
    # plt.ylabel("sky [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("sky_var.png"), dpi=300.)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(8., 4.))
    # plt.suptitle("PSF measured star magnitude vs sky level")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['m'], stars_df['sky'], ',', color='black')
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("sky [mag]")
    # plt.grid()
    # plt.show()
    # plt.savefig(smphot_stars_output.joinpath("mag_sky.png"), dpi=300.)
    # plt.close()

    # plt.figure(figsize=(5., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['m'], stars_df['mag'], ',', color='black')
    # plt.title("sky")
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$m$ [gaia mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("mag_mag_gaia.png"), dpi=300.)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(8., 4.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['m'], stars_df['em'], ',', color='black')
    # plt.ylim(0., 0.3)
    # plt.xlabel("$m$ [mag]")
    # plt.ylabel("$\\sigma_m$")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("mag_var.png"), dpi=300.)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(8., 5.))
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['mag'], stars_df['res'], ',', color='black')
    # plt.ylim(-1., 1.)
    # plt.xlabel("$m$ [Gaia mag]")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.grid()
    # plt.savefig(smphot_stars_output.joinpath("residuals_mag.png"), dpi=300.)
    # plt.show()
    # plt.close()

    # plt.figure()
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['em'], stars_df['res'], ',', color='black')
    # plt.grid()
    # plt.xlim(0., 0.3)
    # plt.ylim(-0.5, 0.5)
    # plt.xlabel("$\sigma_m$ [mag]")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.show()
    # plt.savefig(smphot_stars_output.joinpath("var_residuals.png"), dpi=300.)
    # plt.close()


    # plt.figure(figsize=(10., 4.))
    # plt.title("Residuals as a function of airmass")
    # ax = plt.gca()
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.plot(stars_df['airmass'], stars_df['res'], ',')
    # plt.ylim(-0.5, 0.5)
    # plt.xlabel("$X$")
    # plt.ylabel("$m-\\left<m\\right>$ [mag]")
    # plt.grid()
    # plt.show()
    # plt.savefig(smphot_stars_output.joinpath("airmass_mag.png"), dpi=300.)
    # plt.close()

    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.), gridspec_kw={'hspace': 0.}, sharex=True)
    plt.suptitle("Standardized residuals")
    ax = plt.subplot(2, 1, 1)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #xbinned_mag, yplot_student_res, res_dispersion = binplot(stars_df['mag'].to_numpy(), (stars_df['res']/stars_df['em']).to_numpy(), nbins=10, data=True, rms=False, scale=False)
    xbinned_mag, yplot_student_res, res_dispersion = binplot(stars_df['mag'].to_numpy(), w*stars_df['res'].to_numpy(), nbins=10, data=True, rms=False, scale=False)
    plt.ylabel("$\\frac{m-\\left<m\\right>}{\\sigma_m}$")
    plt.grid()

    ax = plt.subplot(2, 1, 2)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\frac{m-\\left<m\\right>}{\\sigma_m}$")

    plt.savefig(smphot_stars_output.joinpath("student.png"), dpi=500.)
    plt.show()

    mjd_min, mjd_max = stars_df['mjd'].min(), stars_df['mjd'].max()

    for star_index in list(set(stars_df['star'])):
        star_mask = (stars_df['star'] == star_index)
        outlier_star_mask = (outlier_stars_df['star'] == star_index)
        if sum(star_mask) == 0 or np.any(stars_df.loc[star_mask]['flux'] <= 0.):
            continue

        m = solver.model.params.free[dp.star_map[star_index]]
        fig = plt.subplots(ncols=2, nrows=1, figsize=(12., 4.), gridspec_kw={'width_ratios': [5, 1], 'wspace': 0, 'hspace': 0}, sharey=True)
        plt.suptitle("Star {} - Gaia mag={}".format(star_index, stars_df.loc[star_mask]['mag'].tolist()[0]))
        ax = plt.subplot(1, 2, 1)
        plt.xlim(mjd_min, mjd_max)
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # plt.errorbar(stars_df.loc[star_mask]['mjd'], stars_df.loc[star_mask]['flux'], yerr=stars_df.loc[star_mask]['error'], marker='.', color='black', ls='')
        plt.errorbar(stars_df.loc[star_mask]['mjd'], stars_df.loc[star_mask]['m'], yerr=stars_df.loc[star_mask]['em'], marker='.', color='black', ls='')
        if len(outlier_star_mask) > 0.:
            plt.errorbar(outlier_stars_df.loc[outlier_star_mask]['mjd'], outlier_stars_df.loc[outlier_star_mask]['m'], yerr=outlier_stars_df.loc[outlier_star_mask]['em'], marker='x', color='black', ls='')
        plt.axhline(m, color='black')
        plt.grid(linestyle='--', color='black')
        plt.xlabel("MJD")
        plt.ylabel("$m$")
        ax = plt.subplot(1, 2, 2)
        ax.tick_params(which='both', direction='in')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xticklabels([])
        s = stars_df.loc[star_mask]['res'].std()
        x = np.linspace(stars_df.loc[star_mask]['m'].min(), stars_df.loc[star_mask]['m'].max(), 100)
        plt.plot(norm.pdf(x, loc=m, scale=s), x)
        plt.hist(stars_df.loc[star_mask]['m'], orientation='horizontal', density=True, bins='auto', histtype='step')

        plt.savefig(star_lc_folder.joinpath("star_{}.png".format(star_index)), dpi=250.)
        plt.close()
