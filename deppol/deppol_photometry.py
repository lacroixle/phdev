#!/usr/bin/env python3

import time

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, dia_array
from sksparse import cholmod
from saunerie.fitparameters import FitParameters

from utils import ListTable


class ZPModel():
    def __init__(self, dp, degree=3):
        self.dp = dp
        self.degree = degree
        self.params = self.init_params()

    def init_params(self):
        exposure_count = len(self.dp.exposure_set)
        color_coeffs = [('k_{}'.format(k+1), 1) for k in range(self.degree)]
        fp = FitParameters([*color_coeffs, ('zp', exposure_count)])
        return fp

    def sigma(self, pedestal=0.):
        return np.sqrt(self.dp.emag**2+self.dp.cat_emag**2+pedestal**2)

    @property
    def W(self):
        return dia_array((1./self.sigma()**2, 0), shape=(len(self.dp.nt), len(self.dp.nt)))

    def __call__(self, p, jac=False):
        """
        """
        self.params.free = p

        zp = self.params['zp'].full[self.dp.exposure_index]
        col = self.dp.colormag
        # col = self.dp.bpmag - self.dp.rpmag
        k_i = [self.params['k_{}'.format(k+1)].full[0] for k in range(self.degree)]

        # v = k * col + zp
        v = zp
        for i, k in enumerate(k_i):
            v += k*col**(i+1)

        if not jac:
            return v

        n = len(self.params.free)
        N = len(self.dp.nt) # size of data vector
        i = np.arange(N)

        # elements of the (sparamse) jacobian
        ii, jj, vv = [], [], []

        # dmdk
        for k in range(self.degree):
            ii.append(i)
            jj.append(np.full(N, self.params['k_{}'.format(k+1)].indexof(0)))
            vv.append(col**(k+1))

        # dmdzp
        ii.append(i)
        jj.append(self.params['zp'].indexof(self.dp.exposure_index))
        vv.append(np.full(N, 1.))

        ii = np.hstack(ii)
        jj = np.hstack(jj)
        vv = np.hstack(vv)
        ok = jj>=0

        J = coo_matrix((vv[ok], (ii[ok], jj[ok])), shape=(N, n))

        return v,J


def _fit_photometry(model, logger):
    logger.info("Photometric fit...")
    t = time.perf_counter()
    p = model.params.free.copy()
    v, J = model(p, jac=1)
    H = J.T @ model.W @ J
    B = J.T @ model.W @ (model.dp.mag - model.dp.cat_mag)
    fact = cholmod.cholesky(H.tocsc())
    p = fact(B)
    model.params.free = p
    logger.info("Done. Elapsed time={}.".format(time.perf_counter()-t))
    return p


def _filter_noisy_stars(model, y_model, threshold, logger):
    logger.info("Removing noisy stars...")
    res = y_model - (model.dp.mag - model.dp.cat_mag)
    chi2 = np.sqrt(np.bincount(model.dp.catid_index, weights=res**2)/np.bincount(model.dp.catid_index))

    logger.info("Chi2 treshold={}".format(threshold))
    noisy_stars = model.dp.catid_set[chi2 > threshold]
    noisy_measurements = np.any([model.dp.catid == noisy_star for noisy_star in noisy_stars], axis=0)

    model.dp.compress(~noisy_measurements)
    logger.info("Filtered {} stars...".format(len(noisy_stars)))
    return ZPModel(model.dp)


def _dump_photoratios(model, y_model, reference_exposure, save_folder_path):
    zp_ref = model.params['zp'].full[model.dp.exposure_map[reference_exposure]]

    alphas = {}
    for exposure in model.dp.exposure_set:
        alphas[exposure] = 10**(-0.4*(model.params['zp'].full[model.dp.exposure_map[exposure]] - zp_ref))

    # TODO: compute error on alpha
    alphas_df = pd.DataFrame(data={'expccd': list(alphas.keys()), 'alpha': list(alphas.values())})
    alphas_df['ealpha'] = 0.

    ndof = len(y_model) - len(model.params.free)
    #chi2 = np.sum((y_model - (model.dp.mag - model.dp.gaia_mag))**2/y_model)
    chi2 = np.sum((y_model - (model.dp.mag - model.dp.cat_mag))**2/np.sqrt(model.dp.emag**2+model.dp.cat_emag**2))

    photom_ratios_table = ListTable({'CHI2': chi2, 'NDOF': ndof, 'RCHI2': chi2/ndof, 'REF': reference_exposure}, alphas_df)
    photom_ratios_table.write_to(save_folder_path.joinpath("photom_ratios.ntuple"))


def photometry_fit(lightcurve, logger, args):
    import pandas as pd
    from croaks import DataProxy
    from utils import filtercode2extcatband, make_index_from_list, get_ref_quadrant_from_band_folder
    import pickle

    save_folder_path = lightcurve.photometry_path
    save_folder_path.mkdir(exist_ok=True)
    lightcurve.mappings_path.mkdir(exist_ok=True)

    logger.info("Building DataProxy")

    # Build a dataproxy from measures and augment it
    matched_stars_df = lightcurve.extract_star_catalog(['psfstars', args.photom_cat])
    matched_stars_df.to_csv("out.csv")

    exposures_df = lightcurve.extract_exposure_catalog()
    for column in exposures_df.columns:
        matched_stars_df[column] = exposures_df.loc[matched_stars_df['exposure'], column].to_numpy()

    matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['psfstars_flux'])
    matched_stars_df['emag'] = 1.08*matched_stars_df['psfstars_eflux']/matched_stars_df['psfstars_flux']
    matched_stars_df['cat_mag'] = matched_stars_df['{}_{}'.format(args.photom_cat, filtercode2extcatband[args.photom_cat][lightcurve.filterid])]
    matched_stars_df['cat_emag'] = matched_stars_df['{}_e_{}'.format(args.photom_cat, filtercode2extcatband[args.photom_cat][lightcurve.filterid])]

    if args.photom_cat == 'gaia':
        matched_stars_df['colormag'] = matched_stars_df['gaia_BPmag'] - matched_stars_df['gaia_RPmag']
        matched_stars_df.rename(columns={'gaia_Source': 'catid'}, inplace=True)
        # Filter out non linear response
        matched_stars_df = matched_stars_df.loc[(matched_stars_df['colormag'] > 0.5)]
        matched_stars_df = matched_stars_df.loc[(matched_stars_df['colormag'] < 1.5)]
    elif args.photom_cat == 'ps1':
        matched_stars_df['colormag'] = matched_stars_df['ps1_gmag'] - matched_stars_df['ps1_zmag']
        matched_stars_df.rename(columns={'ps1_objID': 'catid'}, inplace=True)
        matched_stars_df = matched_stars_df.loc[(matched_stars_df['colormag'] > 0.5)]

    matched_stars_df = matched_stars_df[['mag', 'emag', 'cat_mag', 'cat_emag', 'colormag', 'exposure', 'catid', 'mjd', 'rcid', 'airmass']]
    matched_stars_df.dropna(inplace=True)


    kwargs = dict([(keyword, keyword) for keyword in matched_stars_df.columns])

    dp_index_list = ['exposure', 'catid', 'rcid', 'mjd']
    dp = DataProxy(matched_stars_df.to_records(), **kwargs)
    make_index_from_list(dp, dp_index_list)

    model = ZPModel(dp)

    _fit_photometry(model, logger)
    y_model = model(model.params.free)

    with open(save_folder_path.joinpath("model.pickle"), 'wb') as f:
        pickle.dump(model, f)

    new_model = _filter_noisy_stars(model, y_model, args.photom_max_star_chi2, logger)

    _fit_photometry(new_model, logger)
    y_new_model = new_model(new_model.params.free)

    with open(save_folder_path.joinpath("filtered_model.pickle"), 'wb') as f:
        pickle.dump(new_model, f)

    logger.info("Computing photometric ratios")
    _dump_photoratios(new_model, y_new_model, lightcurve.get_reference_exposure(), lightcurve.mappings_path)

    return True


def photometry_fit_plot(lightcurve, logger, args):
    import pickle
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from saunerie.plottools import binplot
    matplotlib.use('Agg')

    from utils import idx2markerstyle

    def _do_plots(model, save_folder=None, plot_ext=".png"):
        y_model = model(model.params.free)
        def _show(filename):
            if save_folder is not None:
                plt.savefig(save_folder.joinpath("{}{}".format(filename, plot_ext)), dpi=250.)
            else:
                plt.show()

            plt.close()

        k_i = [model.params['k_{}'.format(k+1)].full[0] for k in range(model.degree)]
        zp = model.params['zp'].full[:]
        res = y_model - (model.dp.mag - model.dp.cat_mag)
        chi2 = np.bincount(model.dp.catid_index, weights=res**2)/np.bincount(model.dp.catid_index)
        measurement_chi2 = np.array([chi2[model.dp.catid_map[catid]] for catid in model.dp.catid])

        plt.plot([model.dp.catid_map[catid] for catid in model.dp.catid_set], np.sqrt(chi2), '.', color='black')
        plt.grid()
        plt.xlabel("Star index")
        plt.ylabel("RMS Res")
        _show("chi2_star")

        plt.plot(model.dp.mag, 1./model.sigma()**2, '.', color='black')
        plt.xlabel("$m$")
        plt.ylabel("$1/\\sigma**2$")
        plt.grid()
        _show("sigma_mag")

        rms = [np.std(model.dp.mag[model.dp.catid==catid]-model.params['zp'].full[model.dp.exposure_index[model.dp.catid==catid]], ddof=1) for catid in model.dp.catid_set]

        plt.plot(range(len(model.dp.catid_set)), rms, '.', color='xkcd:light blue')
        plt.xlabel("Cat star #")
        plt.ylabel("Lightcurve RMS")
        plt.grid()
        _show("rms_star")

        # plt.plot(model.dp.cat_mag_set, rms, '.', color='xkcd:light blue')
        # plt.xlabel("$g_G$")
        # plt.ylabel("Lightcurve RMS")
        # plt.grid()
        # _show("rms_mag")

        # bins=50
        # pull_hist = np.histogram2d(model.dp.ra, model.dp.dec, weights=res/model.sigma(), bins=bins)[0]/np.histogram2d(model.dp.ra, model.dp.dec, weights=1./model.sigma(), bins=bins)[0]
        # plt.imshow(pull_hist)
        # plt.xlabel("ra")
        # plt.ylabel("dec")
        # _show("pull_radec")

        plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(10., 5.))
        pull = res/model.sigma(pedestal=0.01)
        m = np.mean(pull)
        s = np.std(pull)
        n = 4. # Limit multiplier for more consistant plots

        # plt.subplot(1, 2, 1)
        # #plt.plot(model.dp.mag, pull, ',', color='black')
        # for i, rcid in enumerate(model.dp.rcid_set):
        #     mask = (model.dp.rcid == rcid)
        #     plt.scatter(model.dp.mag[mask], pull[mask], c=measurement_chi2[mask], s=0.05, marker=idx2markerstyle[i], label="{}".format(rcid))
        # cbar = plt.colorbar()
        # cbar.set_label("$\\chi^2_p$")
        # plt.legend()
        # plt.ylim(-n*s, n*s)
        # plt.xlabel("m")
        # plt.ylabel("pull")
        # plt.grid()

        # ax = plt.subplot(1, 2, 2)
        # plt.hist(pull, range=[-n*s, n*s], bins=50, orientation='horizontal', histtype='step', color='black', density=True)
        # x = np.linspace(-n*s, n*s, 100)
        # plt.plot(norm.pdf(x, loc=m, scale=s), x)
        # plt.text(0., -0.1, "$\\sigma={:.5f}, \\mu={:.5f}$".format(s, m), transform=ax.transAxes)
        # plt.xticks([])
        # plt.yticks([])
        # _show("pull_mag")

        # xbinned_mag, yplot_pull, pull_dispersion = binplot(model.dp.mag, pull, data=True, rms=True, scale=False)
        # plt.xlabel("$m$")
        # plt.ylabel("pull")
        # _show("pull_profile")

        # plt.plot(xbinned_mag, pull_dispersion, color='black')
        # plt.grid()
        # plt.xlabel("m")
        # plt.ylabel("pull RMS")
        # _show("pull_rms")

        # for rcid in model.dp.rcid_set:
        #     mask = model.dp.rcid == rcid
        #     plt.hist2d(model.dp.x[mask], model.dp.y[mask], weights=res[mask], bins=100, vmin=-0.5, vmax=0.5)
        #     plt.title("Residuals, rcid={}, n_measurements={}".format(rcid, len(res[mask])))
        #     plt.axis('equal')
        #     plt.axis('off')
        #     plt.xlabel("$x$")
        #     plt.ylabel("$y$")
        #     plt.colorbar()
        #     _show("quad_{}".format(rcid))

        # ZP distribution
        plt.hist(zp, bins=40, histtype='step', color='black')
        plt.grid()
        plt.xlabel("ZP")
        plt.ylabel("Count")
        _show("zp_dist")

        # Residuals distribution
        plt.hist(res, bins=100, histtype='step', color='black')
        plt.grid()
        plt.xlabel("Residuals")
        plt.ylabel("Count")
        _show("res_dist")

        # Residuals/day
        plt.plot(model.dp.mjd, res, ',', color='red')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("Residual")
        _show("res_day")

        plt.plot(model.dp.mjd_index, res, ',', color='red')
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("Residual")
        _show("res_day_index")

        #Residuals/star
        plt.plot(model.dp.catid_index, res, ',', color='red')
        plt.grid()
        plt.xlabel("Star")
        plt.ylabel("Residual")
        _show("res_star")

        # Residuals/airmass
        plt.plot(model.dp.airmass, res, ',', color='red')
        plt.xlabel('Airmass')
        plt.ylabel('Residual')
        plt.grid()
        _show("res_airmass")

        # Residuals/color
        plt.plot(model.dp.colormag, res, ',', color='red')
        plt.xlabel("$B_p-R_p$")
        plt.ylabel("Residual")
        plt.grid()
        _show("res_color")

        # Residuals/mag
        plt.plot(model.dp.mag, res, ',', color='red')
        plt.xlabel("$m$")
        plt.ylabel("Residual")
        plt.grid()
        _show("res_mag")

    save_folder_path = lightcurve.photometry_path

    # First do residuals plot of the relative photometry models (non filtered and filtered)
    with open(save_folder_path.joinpath("model.pickle"), 'rb') as f:
        model = pickle.load(f)

    with open(save_folder_path.joinpath("filtered_model.pickle"), 'rb') as f:
        filtered_model = pickle.load(f)

    plot_no_filter_folder = save_folder_path.joinpath("no_filter")
    plot_filter_folder = save_folder_path.joinpath("filter")

    plot_no_filter_folder.mkdir(exist_ok=True)
    plot_filter_folder.mkdir(exist_ok=True)

    _do_plots(model, save_folder=plot_no_filter_folder)
    _do_plots(filtered_model, save_folder=plot_filter_folder)
