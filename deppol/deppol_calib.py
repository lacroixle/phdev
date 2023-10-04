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

    df = pd.read_parquet(lightcurve.smphot_stars_path.joinpath("constant_stars.parquet"))
    df['istar'] = df.index
    df['delta_mag'] = df['m'] - df['cat_mag']
    df.to_csv("constant_stars.csv")

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

    res = solver.get_res(dp.delta_mag)
    wres = res/dp.delta_mage
    chi2 = np.bincount(dp.istar_index, weights=wres**2)/np.bincount(dp.istar_index)

    plot_path = lightcurve.path.joinpath("calib")
    plot_path.mkdir(exist_ok=True)

    plt.subplots(ncols=1, nrows=1, figsize=(5., 4.))
    plt.plot(chi2, '.', color='black')
    plt.axhline(1.)
    plt.grid()
    plt.xlabel("Star ID")
    plt.ylabel("$\chi^2$")
    plt.show()
    plt.savefig(plot_path.joinpath("chi2.png"), dpi=200.)
    plt.close()

    plt.subplots(ncols=1, nrows=1, figsize=(7., 4.))
    plt.plot(dp.cat_mag[~solver.bads], res, '.', color='black')
    plt.grid()
    plt.xlabel("$m$ [AB mag]")
    plt.ylabel("$y-y_\mathrm{model}$ [mag]")
    plt.savefig(plot_path.joinpath("res.png"), dpi=200.)
    plt.close()


    update_yaml(lightcurve.path.joinpath("lightcurve.yaml"), 'calib',
                {'color': solver.model.params['color'].full[0].item(),
                 'zp': solver.model.params['zp'].full[0].item(),
                 'cov': solver.get_cov().todense().tolist(),
                 'chi2': np.sum(wres**2).item(),
                 'ndof': len(dp.nt[~solver.bads]),
                 'chi2/ndof': np.sum(wres**2).item()/len(dp.nt[~solver.bads])})

    return True
