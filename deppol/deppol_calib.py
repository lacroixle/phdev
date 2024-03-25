#!/usr/bin/env python3


def calib(lightcurve, logger, args):
    from deppol_utils import update_yaml
    from utils import mag2extcatmag, emag2extcatemag, get_ubercal_catalog_in_cone
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic
    from croaks.match import NearestNeighAssoc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from saunerie.plottools import binplot
    import shutil
    from scipy.stats import norm
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    from lightcurve import Exposure

    # matplotlib.use('Agg')

    if not lightcurve.smphot_stars_path.joinpath("constant_stars.parquet").exists():
        logger.error("No constant stars catalog!")
        return False

    # Load constant stars catalog, Gaia catalog (for star identification/matching) and external calibration catalog
    stars_df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    gaia_df = lightcurve.get_ext_catalog('gaia').set_index('Source', drop=False).loc[stars_df.index]
    ps1_df = lightcurve.get_ext_catalog('ps1')
    # ext_cat_df = lightcurve.get_ext_catalog(args.photom_cat).loc[gaia_df['index']]
    ext_cat_df = lightcurve.get_ext_catalog(args.photom_cat)

    reference_exposure = Exposure(lightcurve, lightcurve.get_reference_exposure())
    # ext_cat_df = get_ubercal_catalog_in_cone('repop', args.ubercal_config_path, reference_exposure.center()[0], reference_exposure.center()[1], 0.6, filtercode=lightcurve.filterid)
    # ext_cat_df = get_ubercal_catalog_in_cone('repop', args.ubercal_config_path, reference_exposure.center()[0], reference_exposure.center()[1], 0.6)

    # ext_cat_df = ext_cat_df.join(gaia_df, lsuffix='ext', rsuffix='gaia', how='inner')
    # print(ext_cat_df[['raext', 'decext', 'ragaia', 'decgaia']])

    # plt.subplot(1, 2, 1)
    # plt.hist(3600*(ext_cat_df['raext']-ext_cat_df['ragaia']), bins='auto')
    # plt.xlabel("RA_Uber - RA_Gaia [arcsec]")
    # plt.ylabel("Count")

    # plt.subplot(1, 2, 2)
    # plt.hist(3600*(ext_cat_df['decext']-ext_cat_df['decgaia']), bins='auto')
    # plt.xlabel("DEC_Uber - DEC_Gaia [arsec]")
    # plt.savefig("hist.png", dpi=200.)
    # plt.close()
    # return True

    # ext_cat_df = ext_cat_df.loc[ext_cat_df['calflux_weighted_mean']>0.]
    # ext_cat_df['calflux_rms'] = ext_cat_df['calflux_weighted_std']
    # ext_cat_df['calflux_weighted_std'] = ext_cat_df['calflux_weighted_std']/np.sqrt(ext_cat_df['n_obs']-1)
    # ext_cat_df.rename(columns={'calmag_weighted_mean': '{}mag'.format(lightcurve.filterid), 'calmag_weighted_std': 'e{}mag'.format(lightcurve.filterid), 'calmag_rms': '{}rms'.format(lightcurve.filterid), 'n_obs': '{}_n_obs'.format(lightcurve.filterid), 'chi2_Source_res': '{}_chi2_Source_res'.format(lightcurve.filterid)}, inplace=True)

    # import matplotlib.pyplot as plt
    # plt.plot(ext_cat_df['ra'], ext_cat_df['dec'], 'x', color='grey')
    # # plt.plot(ps1_df['ra'], ps1_df['dec'], '+')
    # plt.plot(gaia_df['ra'], gaia_df['dec'], '.')

    if len(ext_cat_df) == 0:
        logger.error("Empty calibration catalog \'{}\'!".format(args.photom_cat))
        return False

    # Match external catalog with constant star catalog
    assoc = NearestNeighAssoc(first=[np.deg2rad(gaia_df['ra'].to_numpy()), np.deg2rad(gaia_df['dec'].to_numpy())], radius = np.deg2rad(2./60./60.))
    i = assoc.match(np.deg2rad(ext_cat_df['ra'].to_numpy()), np.deg2rad(ext_cat_df['dec'].to_numpy()))

    gaia_df = gaia_df.iloc[i[i>=0]].reset_index(drop=True)
    ext_cat_df = ext_cat_df.iloc[i>=0].reset_index(drop=True)
    print(ext_cat_df)

    plt.plot(gaia_df['ra'], gaia_df['dec'], '.', color='black')
    plt.show()
    plt.savefig("catalogs.png", dpi=1000.)
    plt.close()
    print(ext_cat_df)

    return True

    stars_df = stars_df.loc[gaia_df['Source'].tolist()]

    # Add matched band external catalog magnitude, delta and color
    #piedestal = 0.05 # For ps1
    piedestal = 0.
    stars_df = stars_df.assign(cat_mag=ext_cat_df[mag2extcatmag[args.photom_cat][lightcurve.filterid]].tolist(),
                               cat_emag=ext_cat_df[emag2extcatemag[args.photom_cat][lightcurve.filterid]].tolist())
    stars_df = stars_df.assign(delta_mag=(stars_df['mag'] - stars_df['cat_mag']),
                               delta_emag=np.sqrt(stars_df['emag']**2+stars_df['cat_emag']**2+piedestal**2))
    stars_df = stars_df.assign(cat_color=(ext_cat_df[mag2extcatmag[args.photom_cat]['zg']]-ext_cat_df[mag2extcatmag[args.photom_cat]['zi']]).tolist(),
                               cat_ecolor=np.sqrt(ext_cat_df[emag2extcatemag[args.photom_cat]['zg']]**2+ext_cat_df[emag2extcatemag[args.photom_cat]['zi']]**2).tolist())
    stars_df = stars_df.assign(gaia_color=gaia_df['BP-RP'])

    stars_df.dropna(subset=['cat_emag', 'cat_ecolor'], inplace=True)

    stars_df = stars_df.loc[stars_df['cat_color']<2.5]
    stars_df = stars_df.loc[stars_df['cat_color']>0.]

    stars_df.reset_index(inplace=True)

    # Remove nans and infs
    stars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stars_df.dropna(subset=['delta_mag', 'delta_emag'], inplace=True)

    w = 1./stars_df['delta_emag'].to_numpy()

    dp = DataProxy(stars_df[['delta_mag', 'delta_emag', 'cat_mag', 'cat_emag', 'gaiaid', 'cat_color', 'gaia_color']].to_records(), delta_mag='delta_mag', delta_emag='delta_emag', cat_mag='cat_mag', cat_emag='cat_emag', gaiaid='gaiaid', cat_color='cat_color', gaia_color='gaia_color')
    dp.make_index('gaiaid')

    def _build_model(dp):
        model = indic(np.zeros(len(dp.nt), dtype=int), val=dp.cat_color, name='color') + indic(np.zeros(len(dp.nt), dtype=int), name='zp')
        return RobustLinearSolver(model, dp.delta_mag, weights=w)

    def _solve_model(solver):
        solver.model.params.free = solver.robust_solution(local_param='color')

    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.delta_mag)[~solver.bads]
    wres = res/dp.delta_emag[~solver.bads]

    # Extract fitted parameters
    ZP = solver.model.params['zp'].full.item()
    alpha = solver.model.params['color'].full.item()

    chi2ndof = np.sum(wres**2)/(len(dp.nt)-2-sum(solver.bads)) # 2 parameters in the model

    dp.compress(~solver.bads)

    # Compute residual dispertion of bright stars
    bright_stars_threshold = 18.
    bright_stars_mask = (dp.cat_mag<=bright_stars_threshold)
    bright_stars_mu, bright_stars_std = np.mean(res[bright_stars_mask]), np.std(res[bright_stars_mask])

    output_path = lightcurve.path.joinpath("calib.{}".format(args.photom_cat))
    output_path.mkdir(exist_ok=True)

    # Chromaticity effect plot
    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nColor term of the {} catalog\n $\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, args.photom_cat, chi2ndof))
    plt.errorbar(dp.cat_color, dp.delta_mag-ZP, yerr=dp.delta_emag, fmt='.')
    plt.plot([np.min(dp.cat_color), np.max(dp.cat_color)], [alpha*np.min(dp.cat_color), alpha*np.max(dp.cat_color)], label="$\\alpha C_\\mathrm{{{}}}, \\alpha=${:.4f}".format(args.photom_cat, alpha))
    plt.legend()
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP$ [mag]".format(args.photom_cat))
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res_chromaticity.png"), dpi=200.)
    plt.close()

    # magerr = np.sqrt(dp.emag**2+(alpha*dp.cat_ecolor)**2)
    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot as a function of star color\n$\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, chi2ndof))
    plt.errorbar(dp.cat_color, res, yerr=dp.delta_emag, fmt='.')
    plt.grid()
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{cat}}}-ZP-\\alpha C_\\mathrm{{{cat}}}$ [mag]".format(cat=args.photom_cat))
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res_color.png"), dpi=200.)
    plt.close()

    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot for the calibration fit onto {}\n$ZP$={:.4f}, $\chi^2/\mathrm{{ndof}}={:.4f}$, Star count={}".format(lightcurve.name, lightcurve.filterid, args.photom_cat, ZP, chi2ndof, len(stars_df)))
    xmin, xmax = np.min(dp.cat_mag)-0.2, np.max(dp.cat_mag)+0.2
    plt.errorbar(dp.cat_mag, res, yerr=dp.delta_emag, fmt='.')
    # plt.scatter(dp.cat_mag, res, c=dp.cat_color, zorder=10., s=6.)
    # plt.colorbar()
    plt.fill_between([xmin, bright_stars_threshold], [bright_stars_mu-bright_stars_std]*2, [bright_stars_mu+bright_stars_std]*2, color='xkcd:sky blue', alpha=0.4, label='Bright stars - $\sigma_\mathrm{{res}}={:.4f}$'.format(bright_stars_std))
    plt.xlim(xmin, xmax)
    plt.ylim(-0.2, 0.2)
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C$ [mag]".format(args.photom_cat))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.joinpath("res.png"), dpi=200.)
    plt.close()


    # Binplot of the residuals
    plt.subplots(ncols=1, nrows=2, figsize=(12., 8.), sharex=True, gridspec_kw={'hspace': 0.})
    plt.suptitle("{}-{}\nResidual plot for the calibration fit onto {}\n$ZP$={:.4f}, $\chi^2/\mathrm{{ndof}}={:.4f}$, Star count={}".format(lightcurve.name, lightcurve.filterid, args.photom_cat, ZP, chi2ndof, len(stars_df)))

    ax = plt.subplot(2, 1, 1)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.cat_mag, res, nbins=10, data=False, scale=False)
    plt.plot(dp.cat_mag, res, '.', color='black', zorder=-10.)
    plt.ylabel("$m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C$ [mag]".format(args.photom_cat))
    plt.ylim(-0.4, 0.4)
    plt.grid()

    ax = plt.subplot(2, 1, 2)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.plot(xbinned_mag, res_dispersion)
    plt.grid()
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$\sigma_{{m_\mathrm{{ZTF}}-m_\mathrm{{{}}}-ZP-\\alpha C}}$ [mag]".format(args.photom_cat))

    plt.tight_layout()
    plt.savefig(output_path.joinpath("binned_res.png"), dpi=200.)
    plt.close()

    plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
    plt.suptitle("{}-{}\nStandardized residuals for the calibration, wrt star magnitude\npiedestal={}".format(lightcurve.name, lightcurve.filterid, piedestal))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_mag, res/dp.delta_emag, data=False, scale=False, nbins=5)
    plt.plot(dp.cat_mag, wres, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.xlim([np.min(dp.cat_mag), np.max(dp.cat_mag)])
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.hist(wres, bins='auto', orientation='horizontal', density=True)
    m, s = norm.fit(wres)
    x = np.linspace(np.min(wres)-0.5, np.max(wres)+0.5, 200)
    plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
    plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
    plt.legend()
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlim([np.min(dp.cat_mag), np.max(dp.cat_mag)])
    plt.xlabel("$m_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.axhline(1.)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.joinpath("pull_mag.png"), dpi=200.)
    plt.close()

    plt.subplots(nrows=2, ncols=2, figsize=(10., 6.), gridspec_kw={'width_ratios': [5., 1.5], 'hspace': 0., 'wspace': 0.}, sharex=False, sharey=False)
    plt.suptitle("{}-{}\nStandardized residuals for the calibration, wrt star color\npiedestal={}".format(lightcurve.name, lightcurve.filterid, piedestal))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_color, wres, data=False, scale=False, nbins=5)
    plt.plot(dp.cat_color, wres, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.xlim([np.min(dp.cat_color), np.max(dp.cat_color)])
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.hist(wres, bins='auto', orientation='horizontal', density=True)
    m, s = norm.fit(wres)
    x = np.linspace(np.min(wres)-0.5, np.max(wres)+0.5, 200)
    plt.plot(norm.pdf(x, loc=m, scale=s), x, label="$\sim\mathcal{{N}}(\mu={:.2f}, \sigma={:.2f})$".format(m, s))
    plt.plot(norm.pdf(x, loc=0., scale=1.), x, label="$\sim\mathcal{N}(\mu=0, \sigma=1)$")
    plt.legend()
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlabel("$C_\mathrm{{{}}}$ [mag]".format(args.photom_cat))
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.xlim([np.min(dp.cat_color), np.max(dp.cat_color)])
    # plt.axhline(1.)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.joinpath("pull_color.png"), dpi=200.)
    plt.close()

    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'calib',
                {'color': solver.model.params['color'].full[0].item(),
                 'zp': solver.model.params['zp'].full[0].item(),
                 'cov': solver.get_cov().todense().tolist(),
                 'chi2': np.sum(wres**2).item(),
                 'ndof': len(dp.nt),
                 'chi2/ndof': np.sum(wres**2).item()/(len(dp.nt)-2),
                 'outlier_count': np.sum(solver.bads).item(),
                 'piedestal': piedestal,
                 'bright_stars_res_std': bright_stars_std.item(),
                 'bright_stars_res_mu': bright_stars_mu.item(),
                 'bright_stars_threshold': bright_stars_threshold,
                 'bright_stars_count': len(dp.nt[bright_stars_mask])})

    shutil.copy(lightcurve.path.joinpath("lightcurve.yaml"), lightcurve.path.joinpath("lightcurve_{}-{}.yaml".format(lightcurve.name, lightcurve.filterid)))

    return True
