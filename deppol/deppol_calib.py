#!/usr/bin/env python3


def calib(lightcurve, logger, args):
    from utils import ListTable
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from saunerie.linearmodels import LinearModel, RobustLinearSolver, indic
    import numpy as np
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

    print(solver.model.params)

    res = solver.get_res(dp.delta_mag)
    wres = res/dp.delta_mage
    chi2 = np.bincount(dp.star_index, weights=wres**2)/np.bincount(dp.star_index)



    plt.plot(chi2, '.')
    plt.axhline(1.)
    plt.show()

    plt.plot(dp.mag, res, '.')
    plt.show()

    with open(lightcurve.path.joinpath("zp"), 'w') as f:
        f.write(str(solver.model.params['zp'].full[0]))

    return True
