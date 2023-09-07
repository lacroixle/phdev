#!/usr/bin/env python3


def calib(lightcurve, logger, args):
    from utils import ListTable
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic
    import numpy as np
    import yaml
    from utils import make_index_from_list

    df = ListTable.from_filename(lightcurve.smphot_stars_path.joinpath("smphot_stars_cat.list"), delim_whitespace=False).df
    df = df.loc[df['flux']>0.]
    df['smp_mag'] = -2.5*np.log10(df['flux'])
    df['smp_mage'] = 1.08*df['error']/df['flux']

    df['colormag'] = df['g'] - df['i']
    df['delta_mag'] = df['smp_mag'] - df['mag']

    piedestal = 0.015
    df['delta_mage'] = np.sqrt(df['smp_mage']**2+df['mage']**2+piedestal**2)

    df = df[['smp_mag', 'smp_mage', 'colormag', 'delta_mag', 'delta_mage', 'star', 'mjd', 'mag', 'mage', 'name']]

    kwargs = dict([(keyword, keyword) for keyword in df.columns])
    dp_index_list = ['name', 'star', 'mjd', 'mag', 'colormag']

    dp = DataProxy(df.to_records(), **kwargs)
    make_index_from_list(dp, dp_index_list)

    def _build_model(dp):
        model = indic(np.zeros(len(dp.nt), dtype=int), val=dp.colormag, name='color') + indic(np.zeros(len(dp.nt), dtype=int), name='zp')
        return RobustLinearSolver(model, dp.delta_mag, weights=1./dp.delta_mage)

    def _solve_model(solver):
        solver.model.params.free = solver.robust_solution()

    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.delta_mag)
    wres = res/dp.delta_mage
    chi2 = np.bincount(dp.star_index, weights=wres**2)/np.bincount(dp.star_index)
    noisy_stars = dp.star_set[chi2 > 4.]
    noisy_measurements = np.any([dp.star == noisy_star for noisy_star in noisy_stars], axis=0)
    dp.compress(~noisy_measurements)

    solver = _build_model(dp)
    _solve_model(solver)

    res = solver.get_res(dp.delta_mag)[~solver.bads]
    wres = res/dp.delta_mage[~solver.bads]
    chi2 = np.bincount(dp.star_index[~solver.bads], weights=wres**2)/np.bincount(dp.star_index[~solver.bads])

    plot_path = lightcurve.path.joinpath("calib")
    plot_path.mkdir(exist_ok=True)

    print(np.sum(wres**2), len(dp.nt[~solver.bads]), np.sum(wres**2)/len(dp.nt[~solver.bads]))

    plt.subplots(ncols=1, nrows=1, figsize=(5., 4.))
    plt.plot(chi2, '.', color='black')
    plt.axhline(1.)
    plt.grid()
    plt.xlabel("Star ID")
    plt.ylabel("$\chi^2$")
    plt.savefig(plot_path.joinpath("chi2.png"), dpi=200.)

    plt.subplots(ncols=1, nrows=1, figsize=(7., 4.))
    plt.plot(dp.mag[~solver.bads], res, ',', color='black')
    plt.grid()
    plt.xlabel("$m$ [AB mag]")
    plt.ylabel("$y-y_\mathrm{model}$ [mag]")
    plt.savefig(plot_path.joinpath("res.png"), dpi=200.)

    lightcurve_yaml = {}
    if lightcurve.path.joinpath("lightcurve.yaml").exists():
        with open(lightcurve.path.joinpath("lightcurve.yaml"), 'r') as f:
            lightcurve_yaml = yaml.load(f, Loader=yaml.Loader)

    if lightcurve_yaml is None:
        lightcurve_yaml = {}

    with open(lightcurve.path.joinpath("lightcurve.yaml"), 'w') as f:
        lightcurve_yaml['calib'] = {'color': solver.model.params['color'].full[0].item(),
                                    'zp': solver.model.params['zp'].full[0].item(),
                                    'cov': solver.get_cov().todense().tolist(),
                                    'chi2': np.sum(wres**2).item(),
                                    'ndof': len(dp.nt[~solver.bads]),
                                    'chi2/ndof': np.sum(wres**2).item()/len(dp.nt[~solver.bads])}

        yaml.dump(lightcurve_yaml, f)

    return True
