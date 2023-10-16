#!/usr/bin/env python3


def calib(lightcurve, logger, args):
    from deppol_utils import update_yaml
    from utils import ListTable, make_index_from_list
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from saunerie.plottools import binplot

    # matplotlib.use('Agg')

    df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    df['istar'] = df.index
    df['delta_mag'] = df['m'] - df['cat_mag']

    piedestal = 0.0
    df['delta_mage'] = np.sqrt(df['em']**2+df['cat_emag']**2+piedestal**2)

    kwargs = dict([(keyword, keyword) for keyword in df.columns])
    dp_index_list = ['istar', 'cat_mag', 'cat_color']

    dp = DataProxy(df.to_records(), **kwargs)
    make_index_from_list(dp, dp_index_list)

    def _build_model(dp):
        model = indic(np.zeros(len(dp.nt), dtype=int), val=dp.cat_color, name='color') + indic(np.zeros(len(dp.nt), dtype=int), name='zp')
        return RobustLinearSolver(model, dp.delta_mag, weights=1./dp.delta_mage)

    def _solve_model(solver):
        solver.model.params.free = solver.robust_solution(local_param='color')

    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.delta_mag)
    wres = res/dp.delta_mage
    chi2 = np.bincount(dp.istar_index, weights=wres**2)/np.bincount(dp.istar_index)
    noisy_stars = dp.istar_set[chi2 > 3.]
    noisy_measurements = np.any([dp.istar == noisy_star for noisy_star in noisy_stars], axis=0)
    dp.compress(~noisy_measurements)

    solver = _build_model(dp)
    _solve_model(solver)

    dp.compress(~solver.bads)

    ZP = solver.model.params['zp'].full.item()
    alpha = solver.model.params['color'].full.item()

    res = solver.get_res(dp.delta_mag)
    wres = res/dp.delta_mage
    chi2 = np.bincount(dp.istar_index, weights=wres**2)/np.bincount(dp.istar_index)

    chi2ndof = np.sum(wres**2)/(len(dp.nt)-2) # 2 parameters in the model

    # Compute residual dispertion of bright stars
    bright_stars_threshold = 18.
    bright_stars_mask = (dp.cat_mag<=bright_stars_threshold)
    bright_stars_mu, bright_stars_std = np.mean(res[bright_stars_mask]), np.std(res[bright_stars_mask])

    plot_path = lightcurve.path.joinpath("calib")
    plot_path.mkdir(exist_ok=True)

    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nChromaticity effect of the PS1 catalog\n $\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, chi2ndof))
    plt.errorbar(dp.cat_color, dp.delta_mag-ZP, yerr=dp.delta_mage, fmt='.')
    plt.plot([np.min(dp.cat_color), np.max(dp.cat_color)], [alpha*np.min(dp.cat_color), alpha*np.max(dp.cat_color)], label="$\\alpha C_\\mathrm{{PS1}}, \\alpha=${:.4f}".format(alpha))
    plt.legend()
    plt.xlabel("$C_\mathrm{PS1}$ [mag]")
    plt.ylabel("$m_\mathrm{ZTF}-m_\mathrm{PS1}-ZP$ [mag]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("calib/res_chromaticity.png"), dpi=200.)
    plt.close()

    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot as a function of star color\n$\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, chi2ndof))
    plt.plot(dp.cat_color, res, '.')
    plt.grid()
    plt.xlabel("$C_\mathrm{PS1}$ [mag]")
    plt.ylabel("$m_\mathrm{ZTF}-m_\mathrm{PS1}-ZP-\alpha C_\mathrm{PS1}$ [mag]")
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("calib/res_color.png"), dpi=200.)
    plt.close()

    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{}-{}\nResidual plot for the calibration fit onto PS1\n$ZP$={:.4f}, $\chi^2/\mathrm{{ndof}}={:.4f}$".format(lightcurve.name, lightcurve.filterid, ZP, chi2ndof))
    xmin, xmax = np.min(dp.cat_mag)-0.2, np.max(dp.cat_mag)+0.2
    plt.plot(dp.cat_mag, res, '.')
    plt.fill_between([xmin, bright_stars_threshold], [bright_stars_mu-bright_stars_std]*2, [bright_stars_mu+bright_stars_std]*2, color='xkcd:sky blue', alpha=0.4, label='Bright stars - $\sigma_\mathrm{{res}}={:.4f}$'.format(bright_stars_std))
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.xlabel("$m_\mathrm{PS1}$ [AB mag]")
    plt.ylabel("$m_\mathrm{ZTF}-m_\mathrm{PS1}-ZP-\\alpha C$ [mag]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("calib/res.png"), dpi=200.)
    plt.close()


    plt.subplots(nrows=2, ncols=1, figsize=(10., 6.), gridspec_kw={'hspace': 0.})
    plt.suptitle("{}-{}\nStandardized residuals for the calibration, wrt star magnitude\npiedestal={}".format(lightcurve.name, lightcurve.filterid, piedestal))
    plt.subplot(2, 1, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_mag, res/dp.delta_mage, data=False, rms=True, scale=False, nbins=5)
    plt.plot(dp.cat_mag, res/dp.delta_mage, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlabel("$m_\mathrm{PS1}$ [AB mag]")
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.axhline(1.)
    plt.grid()
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("calib/pull_mag.png"), dpi=200.)
    plt.close()

    plt.subplots(nrows=2, ncols=1, figsize=(10., 6.), gridspec_kw={'hspace': 0.})
    plt.suptitle("{}-{}\nStandardized residuals for the calibration fit, wrt star color\npiedestal={}".format(lightcurve.name, lightcurve.filterid, piedestal))
    plt.subplot(2, 1, 1)
    xbinned_mag, yplot_stdres, stdres_dispersion = binplot(dp.cat_color, res/dp.delta_mage, data=False, rms=True, scale=False, nbins=5)
    plt.plot(dp.cat_color, res/dp.delta_mage, '.', color='xkcd:light blue')
    plt.ylabel("$\\frac{m-m_\\mathrm{model}}{\\sigma_m}$ [mag]")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(xbinned_mag, stdres_dispersion)
    plt.xlabel("$C_\mathrm{PS1}$ [mag]")
    plt.ylabel("$\\sigma_{\\frac{m-m_\\mathrm{model}}{\\sigma_m}}$ [mag]")
    plt.axhline(1.)
    plt.grid()
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("calib/pull_color.png"), dpi=200.)
    plt.close()

    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'calib',
                {'color': solver.model.params['color'].full[0].item(),
                 'zp': solver.model.params['zp'].full[0].item(),
                 'cov': solver.get_cov().todense().tolist(),
                 'chi2': np.sum(wres**2).item(),
                 'ndof': len(dp.nt),
                 'chi2/ndof': np.sum(wres**2).item()/(len(dp.nt)-2),
                 'outlier_count': np.sum(solver.bads).item(),
                 'bright_stars_res_std': bright_stars_std.item(),
                 'bright_stars_res_mu': bright_stars_mu.item(),
                 'bright_stars_threshold': bright_stars_threshold,
                 'bright_stars_count': len(dp.nt[bright_stars_mask])})

    return True
