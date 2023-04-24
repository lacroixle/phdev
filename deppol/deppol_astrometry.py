#!/usr/bin/env python3

import time

import imageproc.composable_functions as compfuncs
import saunerie.fitparameters as fp
from scipy import sparse
import numpy as np
from sksparse import cholmod

from utils import get_wcs_from_quadrant, quadrant_width_px, quadrant_height_px, gaiarefmjd


def wcs_residuals(lightcurve, logger, args):
    """

    """
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from saunerie.plottools import binplot
    matplotlib.use('Agg')

    logger.info("Generating star catalog")
    matched_stars_df = lightcurve.extract_star_catalog(['psfstars', 'gaia'], project=True)

    matched_stars_df.rename({'gaia_ra': 'ra',
                             'gaia_dec': 'dec',
                             'psfstars_x': 'x',
                             'psfstars_y': 'y',
                             'psfstars_sx': 'sx',
                             'psfstars_sy': 'sy',
                             'gaia_pmRA': 'pmra',
                             'gaia_pmDE': 'pmdec',
                             'gaia_BPmag': 'bpmag',
                             'gaia_RPmag': 'rpmag',
                             'gaia_Source': 'gaiaid',
                             'gaia_Gmag': 'mag',
                             'name': 'exposure'},
                            axis='columns', inplace=True)
    matched_stars_df = matched_stars_df[['exposure', 'gaiaid', 'ra', 'dec', 'x', 'y', 'gaia_x', 'gaia_y', 'sx', 'sy', 'pmra', 'pmdec', 'mag', 'bpmag', 'rpmag']]
    matched_stars_df.dropna(inplace=True)
    logger.info("N={}".format(len(matched_stars_df)))

    save_folder_path= lightcurve.path.joinpath("wcs_residuals_plots")
    save_folder_path.mkdir(exist_ok=True)

    ################################################################################
    # Residuals distribution
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.hist(matched_stars_df['x']-matched_stars_df['gaia_x'], bins=100, range=[-0.5, 0.5])
    plt.grid()
    plt.xlabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.ylabel("#")

    plt.subplot(1, 2, 2)
    plt.hist(matched_stars_df['y']-matched_stars_df['gaia_y'], bins=100, range=[-0.5, 0.5])
    plt.grid()
    plt.xlabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.ylabel("#")

    plt.savefig(save_folder_path.joinpath("wcs_res_dist.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/magnitude
    plt.subplots(nrows=2, ncols=1, figsize=(15., 10.))
    plt.subplot(2, 1, 1)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['x']-matched_stars_df['gaia_x'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmdec']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['y']-matched_stars_df['gaia_y'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmdec']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.savefig(save_folder_path.joinpath("mag_wcs_res.png"), dpi=750.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/magnitude binplot
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(matched_stars_df['mag'].to_numpy(), (matched_stars_df['x']-matched_stars_df['gaia_x']).to_numpy(), nbins=50, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{Gaia}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(matched_stars_df['mag'].to_numpy(), (matched_stars_df['y']-matched_stars_df['gaia_y']).to_numpy(), nbins=50, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{Gaia}}$ [pixel]")

    plt.savefig(save_folder_path.joinpath("mag_wcs_res_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Star lightcurve RMS mag/star lightcurve mean mag
    rms, mean = [], []

    for gaiaid in set(matched_stars_df['gaiaid']):
        gaiaid_mask = (matched_stars_df['gaiaid']==gaiaid)
        rms.append(matched_stars_df.loc[gaiaid_mask, 'mag'].std())
        mean.append(matched_stars_df.loc[gaiaid_mask, 'mag'].mean())

    plt.plot(mean, rms, '.')
    plt.xlabel("$\\left<m\\right>$")
    plt.ylabel("$\\sigma_m$")
    plt.grid()
    plt.savefig(save_folder_path.joinpath("rms_mean_lc.png"), dpi=300.)
    plt.close()
    ################################################################################


class AstromModel():
    """

    """
    def __init__(self, degree, exposure_count):
        self.degree = degree
        self.params = self.init_params(exposure_count)

    def init_params(self, exposure_count):
        self.tp2px = compfuncs.BiPol2D(deg=self.degree, key='exposure', n=exposure_count)
        return fp.FitParameters(self.tp2px.get_struct())

    def sigma(self, dp):
        return np.hstack((dp.sx, dp.sy))

    def W(self, dp):
        return sparse.dia_array((1./self.sigma(dp)**2, 0), shape=(2*len(dp.nt), 2*len(dp.nt)))

    def __call__(self, x, pm, mjd, exposure_indices, jac=False):
        # Correct for proper motion in tangent plane
        dT = mjd - gaiarefmjd
        x = x + pm*dT

        if not jac:
            xy = self.tp2px(x, p=self.params, exposure=exposure_indices)

            return xy
        else:
            # Derivatives wrt polynomial
            xy, df, (i, j, vals) = self.tp2px.derivatives(x, p=self.params, exposure=exposure_indices)

            ii = [np.hstack([i, i+x.shape[1]])]
            jj = [np.tile(j, 2).ravel()]
            vv = [np.hstack(vals).ravel()]

            NN = 2*x.shape[1]
            ii = np.hstack(ii)
            jj = np.hstack(jj)
            vv = np.hstack(vv)
            ok = jj >= 0
            J_model = sparse.coo_array((vv[ok], (ii[ok], jj[ok])), shape=(NN, len(self.params.free)))

            return xy, J_model

    def residuals(self, x, y, pm, mjd, exposure_indices):
        y_model = self.__call__(x, pm, mjd, exposure_indices)
        return y_model - y


def _fit_astrometry(model, dp, logger):
    logger.info("Astrometry fit with {} measurements.".format(len(dp.nt)))

    start_time = time.perf_counter()
    p = model.params.free.copy()
    v, J = model(np.array([dp.tpx, dp.tpy]), np.array([dp.pmtpx, dp.pmtpy]), dp.mjd, dp.exposure_index, jac=True)
    H = J.T @ model.W(dp) @ J
    B = J.T @ model.W(dp) @ np.hstack((dp.x, dp.y))
    fact = cholmod.cholesky(H.tocsc())
    p = fact(B)
    model.params.free = p
    logger.info("Done. Elapsed time={}.".format(time.perf_counter()-start_time))
    return p


def _filter_noisy(model, res_x, res_y, field, threshold, logger):
    """
    Filter elements of defined set whose partial Chi2 is over some threshold
    """

    w = np.sqrt(res_x**2 + res_y**2)
    field_val = getattr(model.dp, field)
    field_idx = getattr(model.dp, '{}_index'.format(field))
    field_set = getattr(model.dp, '{}_set'.format(field))
    chi2 = np.bincount(field_idx, weights=w)/np.bincount(field_idx)

    noisy = field_set[chi2 > threshold]
    noisy_measurements = np.any([field_val == noisy for noisy in noisy], axis=0)

    model.dp.compress(~noisy_measurements)
    logger.info("Filtered {} {}... down to {} measurements".format(len(noisy), field, len(model.dp.nt)))

    return AstromModel(model.dp, degree=model.degree)


def astrometry_fit(lightcurve, logger, args):
    import pickle
    import pandas as pd
    import numpy as np
    from imageproc import gnomonic
    import matplotlib
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from utils import ztf_latitude, BiPol2D_fit, create_2D_mesh_grid, poly2d_to_file, ListTable

    matplotlib.use('Agg')

    lightcurve.astrometry_path.mkdir(exist_ok=True)
    lightcurve.mappings_path.mkdir(exist_ok=True)

    reference_exposure = lightcurve.get_reference_exposure()
    logger.info("Reference exposure: {}".format(reference_exposure))

    sn_parameters_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')

    # Define plot saving folder
    lightcurve.astrometry_path.mkdir(exist_ok=True)

    # Load data
    logger.info("Generating star catalog")
    matched_stars_df = lightcurve.extract_star_catalog(['psfstars', 'gaia'])
    matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['psfstars_flux'])
    matched_stars_df['emag'] = 1.08*matched_stars_df['psfstars_eflux']/matched_stars_df['psfstars_flux']

    # Add exposure informations
    exposures_df = lightcurve.extract_exposure_catalog()
    for column in exposures_df.columns:
        matched_stars_df[column] = exposures_df.loc[matched_stars_df['exposure'], column].to_numpy()

    matched_stars_df.rename({'gaia_ra': 'ra',
                             'gaia_dec': 'dec',
                             'psfstars_x': 'x',
                             'psfstars_y': 'y',
                             'psfstars_sx': 'sx',
                             'psfstars_sy': 'sy',
                             'gaia_pmRA': 'pmra',
                             'gaia_pmDE': 'pmdec',
                             'gaia_BPmag': 'bpmag',
                             'gaia_RPmag': 'rpmag',
                             'gaia_Source': 'gaiaid',
                             'gaia_Gmag': 'cat_mag',
                             'gaia_e_Gmag': 'cat_emag',
                             'name': 'exposure'},
                            axis='columns', inplace=True)
    matched_stars_df = matched_stars_df[['exposure', 'gaiaid', 'ra', 'dec', 'x', 'y', 'sx', 'sy', 'pmra', 'pmdec', 'cat_mag', 'cat_emag', 'mag', 'emag', 'bpmag', 'rpmag', 'mjd', 'seeing', 'z', 'ha', 'airmass', 'rcid']]
    matched_stars_df.dropna(inplace=True)
    matched_stars_df.to_csv("out.csv")
    logger.info("N={}".format(len(matched_stars_df)))
    if args.astro_min_mag:
        logger.info("Filtering out faint stars (magnitude cut={} [mag])".format(args.astro_min_mag))
        matched_stars_df = matched_stars_df.loc[matched_stars_df['mag']<=args.astro_min_mag]
        logger.info("N={}".format(len(matched_stars_df)))

    # Compute parallactic angle
    parallactic_angle_sin = np.cos(np.deg2rad(ztf_latitude))*np.sin(np.deg2rad(matched_stars_df['ha']))/np.sin(np.deg2rad(matched_stars_df['z']))
    parallactic_angle_cos = np.sqrt(1.-parallactic_angle_sin**2)

    # Add paralactic angle
    matched_stars_df['parallactic_angle_x'] = parallactic_angle_sin
    matched_stars_df['parallactic_angle_y'] = parallactic_angle_cos

    # Project to tangent plane from radec degree coordinates
    def _project_tp(ra, dec, ra_center, dec_center, pmra, pmdec):
        tpx, tpy, e_tpx, e_tpy = gnomonic.gnomonic_projection(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra_center), np.deg2rad(dec_center), np.zeros_like(ra), np.zeros_like(ra))

        tpdx, tpdy = gnomonic.gnomonic_dxy(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra_center), np.deg2rad(dec_center))

        tpdx = tpdx.T.reshape(-1, 1, 2)
        tpdy = tpdy.T.reshape(-1, 1, 2)
        pm = np.array([np.deg2rad(pmra), np.deg2rad(pmdec)])
        pm = pm.T.reshape(-1, 2, 1)
        pmtpx = tpdx @ pm
        pmtpy = tpdy @ pm

        return tpx[0], tpy[0], pmtpx.squeeze(), pmtpy.squeeze()

    tpx, tpy, pmtpx, pmtpy = _project_tp(matched_stars_df['ra'].to_numpy(), matched_stars_df['dec'].to_numpy(),
                                         sn_parameters_df['sn_ra'].to_numpy(), sn_parameters_df['sn_dec'].to_numpy(),
                                         matched_stars_df['pmra'].to_numpy(), matched_stars_df['pmdec'].to_numpy())

    matched_stars_df['tpx'] = tpx
    matched_stars_df['tpy'] = tpy
    matched_stars_df['etpx'] = 0.
    matched_stars_df['etpy'] = 0.

    matched_stars_df['pmtpx'] = pmtpx
    matched_stars_df['pmtpy'] = pmtpy
    matched_stars_df['epmtpx'] = 0.
    matched_stars_df['epmtpy'] = 0.

    # Do proper motion correction in tangent plane
    #dT = matched_stars_df['mjd'] - gaiarefmjd
    # matched_stars_df['tpx'] = matched_stars_df['tpx'] + matched_stars_df['pmtpx']*dT
    # matched_stars_df['tpy'] = matched_stars_df['tpy'] + matched_stars_df['pmtpy']*dT

    matched_stars_df['color'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']
    matched_stars_df['centered_color'] = matched_stars_df['color'] - matched_stars_df['color'].mean()

    # Build dataproxy for model
    dp = DataProxy(matched_stars_df.to_records(),
                   x='x', sx='sx', sy='sy', y='y', ra='ra', dec='dec', exposure='exposure', mag='mag', cat_mag='cat_mag', gaiaid='gaiaid', mjd='mjd',
                   bpmag='bpmag', rpmag='rpmag', seeing='seeing', z='z', airmass='airmass', tpx='tpx', tpy='tpy', pmtpx='pmtpx', pmtpy='pmtpy',
                   parallactic_angle_x='parallactic_angle_x', parallactic_angle_y='parallactic_angle_y', color='color',
                   centered_color='centered_color', rcid='rcid', pmra='pmra', pmdec='pmdec')

    dp.make_index('exposure')
    dp.make_index('gaiaid')
    dp.make_index('color')
    dp.make_index('rcid')

    # Compute index of reference quadrant
    reference_index = dp.exposure_map[reference_exposure]

    ################################################################################
    # Tangent space to pixel space

    # Build model
    tp2px_model = AstromModel(args.astro_degree, len(dp.exposure_set))
    # Model fitting
    _fit_astrometry(tp2px_model, dp, logger)

    # Dump proper motion catalog
    def _dump_pm_catalog():
        refmjd = float(lightcurve.exposures[reference_exposure].mjd)
        gaia_stars_df = lightcurve.get_ext_catalog('gaia').rename(columns={'pmRA': 'pmra', 'pmDE': 'pmdec'})

        tpx, tpy, pmtpx, pmtpy = _project_tp(gaia_stars_df['ra'].to_numpy(), gaia_stars_df['dec'].to_numpy(),
                                             sn_parameters_df['sn_ra'].to_numpy(), sn_parameters_df['sn_dec'].to_numpy(),
                                             gaia_stars_df['pmra'].to_numpy(), gaia_stars_df['pmdec'].to_numpy())

        tp = np.array([tpx, tpy])
        pmtp = np.array([pmtpx, pmtpy])

        refxy = tp2px_model(tp, pmtp, [refmjd]*len(gaia_stars_df), [reference_index]*len(gaia_stars_df))
        _, d_tp2px, _ = tp2px_model.tp2px.derivatives(tp, tp2px_model.params, exposure=[reference_index]*len(gaia_stars_df))
        srefxy = np.zeros_like(refxy)
        rhoxy = np.zeros_like(refxy[0])
        flux = np.zeros_like(refxy[0])

        pm = np.einsum("jk,jik->ik", pmtp, d_tp2px)

        plt.subplot(1, 2, 1)
        plt.plot(pm[0], pmtp[0], '.')
        plt.xlabel("$\\mu_x$")
        plt.ylabel("$\\mu_{x_t}$")
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(pm[1], pmtp[1], '.')
        plt.xlabel("$\\mu_y$")
        plt.ylabel("$\\mu_{y_t}$")
        plt.grid()

        plt.savefig(lightcurve.astrometry_path.joinpath("mu_tp_exposure.png"), dpi=300.)
        plt.close()

        plt.hist(np.sqrt(pm[0]**2+pm[1]**2), bins=100, histtype='step')
        plt.xlabel("$\sqrt{\\mu_x^2+\\mu_y^2}$")
        plt.ylabel("Count")
        plt.grid()

        plt.savefig(lightcurve.astrometry_path.joinpath("mu_exposure_dist.png"), dpi=300.)
        plt.close()

        df_dict = {'x': refxy[0], 'y': refxy[1], 'sx': srefxy[0], 'sy': srefxy[1], 'rhoxy': rhoxy, 'flux': flux, 'pmx': pm[0], 'pmy': pm[1]}
        df = pd.DataFrame(data=df_dict)
        df_desc = {'x': "x position (pixels)",
                   'y': "y position (pixels)",
                   'sx': "x position rms",
                   'sy': "y position rms",
                   'rhoxy': "xy correlation",
                   'flux': "flux in image ADUs",
                   'pmx': "proper motion along x (pixels/day)",
                   'pmy': "proper motion along y (pixels/day)"}

        pmcatalog_listtable = ListTable({'REFDATE': refmjd}, df, df_desc, "BaseStar 3 PmStar 1")
        pmcatalog_listtable.write_to(lightcurve.astrometry_path.joinpath("pmcatalog.list"))


    ################################################################################
    # Pixel space to tangent space for reference exposure
    #

    grid_res = 100
    def _ref2tp_polymodel(degree):
        logger.info("Reference exposure={}".format(reference_exposure))
        wcs = lightcurve.exposures[reference_exposure].wcs

        # project corner points to sky then to tangent plane
        logger.info("  Projecting corner points to tangent plane using WCS")
        corner_points_px = np.array([[0., 0.],
                                    [quadrant_width_px, 0.],
                                    [0., quadrant_height_px],
                                    [quadrant_width_px, quadrant_height_px]])
        corner_points_radec = np.vstack(wcs.pixel_to_world_values(corner_points_px)).T

        [corner_points_tpx], [corner_points_tpy], _, _ = gnomonic.gnomonic_projection(np.deg2rad(corner_points_radec[0]), np.deg2rad(corner_points_radec[1]),
                                                                                      np.deg2rad(sn_parameters_df['sn_ra'].to_numpy()), np.deg2rad(sn_parameters_df['sn_dec'].to_numpy()),
                                                                                      np.zeros(4), np.zeros(4))

        grid_points_tp = create_2D_mesh_grid(np.linspace(np.min(corner_points_tpx), np.max(corner_points_tpx), grid_res),
                                             np.linspace(np.min(corner_points_tpy), np.max(corner_points_tpy), grid_res)).T

        grid_points_px = tp2px_model.tp2px(grid_points_tp, tp2px_model.params, exposure=[reference_index]*grid_points_tp.shape[1])

        logger.info("  Fitting using polynomial of degree {}".format(degree))
        lightcurve.astrometry_path.joinpath("ref2tp_plots").mkdir(exist_ok=True)
        ref2tp_model = BiPol2D_fit(grid_points_px, grid_points_tp, degree, control_plots=lightcurve.astrometry_path.joinpath("ref2tp_plots"))

        return ref2tp_model

    logger.info("Computing reference pixel space to tangent space transformation...")
    ref2tp_model = _ref2tp_polymodel(degree=args.astro_degree)

    ################################################################################
    # Reference pixel space to pixel space

    logger.info("Transforming reference pixel space to exposures pixel space")

    grid_res = 25

    # Point grid position in exposure (pixel) space
    grid_points_px = create_2D_mesh_grid(np.linspace(0., quadrant_width_px, grid_res), np.linspace(0., quadrant_height_px, grid_res)).T

    # Reference pixel space to tangent space
    ref_grid_tp = ref2tp_model(grid_points_px)

    # Reference tangent space to quadrant pixel space for each quadrant
    space_indices = np.concatenate([np.full(ref_grid_tp.shape[1], i) for i, _ in enumerate(dp.exposure_set)])
    ref_grid_px = tp2px_model.tp2px(np.tile(np.array([ref_grid_tp[0], ref_grid_tp[1]]), (1, len(dp.exposure_set))), tp2px_model.params, exposure=space_indices)

    ref_idx = dp.exposure_map[reference_exposure]
    ref_mask = (space_indices==ref_idx)

    logger.info("Composing polynomials to get ref -> exposure in pixel space")

    # Polynomials for reference pixel space to quadrant pixel spaces
    lightcurve.astrometry_path.joinpath("ref2px_plots").mkdir(exist_ok=True)
    ref2px_model = BiPol2D_fit(np.tile(grid_points_px, (1, len(dp.exposure_set))), ref_grid_px, args.astro_degree, space_indices,
                               control_plots=lightcurve.astrometry_path.joinpath("ref2px_plots"), simultaneous_fit=False)

    logger.info("Saving coefficients to {}".format(lightcurve.astrometry_path.joinpath("ref2px_coeffs.csv")))
    coeffs_df = pd.DataFrame(data=ref2px_model.coeffs, index=dp.exposure_set)
    coeffs_df.to_csv(lightcurve.astrometry_path.joinpath("ref2px_coeffs.csv"), sep=",")

    logger.info("Writing transformations files to {}".format(lightcurve.mappings_path))
    poly2d_to_file(ref2px_model, dp.exposure_set, lightcurve.mappings_path)

    logger.info("Writing proper motion catalog to {}".format(lightcurve.astrometry_path.joinpath("pmcatalog.list")))
    _dump_pm_catalog()

    logger.info("Saving models to {}".format(lightcurve.astrometry_path.joinpath("models.pickle")))
    with open(lightcurve.astrometry_path.joinpath("models.pickle"), 'wb') as f:
        pickle.dump({'tp2px': tp2px_model, 'ref2tp': ref2tp_model, 'ref2px': ref2px_model, 'dp': dp}, f)

    logger.info("Filtering exposures with bad chi2")


    gaiaid_in_ref = dp.gaiaid[dp.exposure_index == reference_index]
    measure_mask = np.where(dp.exposure_index != reference_index, np.isin(dp.gaiaid, gaiaid_in_ref), False)
    in_ref = np.hstack([np.where(gaiaid == gaiaid_in_ref) for gaiaid in dp.gaiaid[measure_mask]]).flatten()

    ref2px_residuals = ref2px_model.residuals(np.array([dp.x[dp.exposure_index==reference_index][in_ref], dp.y[dp.exposure_index==reference_index][in_ref]]),
                                              np.array([dp.x[measure_mask], dp.y[measure_mask]]),
                                              space_indices=dp.exposure_index[measure_mask])

    ref2px_chi2_exposure = np.bincount(dp.exposure_index[measure_mask], weights=np.sqrt(ref2px_residuals[0]**2+ref2px_residuals[1]**2))/np.bincount(dp.exposure_index[measure_mask])
    ref2px_chi2_exposure[reference_index] = 0.
    ref2px_chi2_exposure = np.pad(ref2px_chi2_exposure, (0, len(dp.exposure_set) - len(ref2px_chi2_exposure)), constant_values=np.nan)

    df_ref2px_chi2_exposure = pd.DataFrame(data=ref2px_chi2_exposure, index=dp.exposure_set, columns=['chi2'])
    df_ref2px_chi2_exposure.to_csv(lightcurve.astrometry_path.joinpath("ref2px_chi2_exposures.csv"), sep=",")
    return True


def astrometry_fit_plot(lightcurve, logger, args):
    import pickle
    from sksparse import cholmod
    from scipy.stats import norm
    from imageproc import gnomonic
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from saunerie.plottools import binplot
    import imageproc.composable_functions as compfuncs
    import saunerie.fitparameters as fp
    from scipy import sparse
    from croaks import DataProxy
    from utils import get_ref_quadrant_from_band_folder

    matplotlib.use('Agg')

    save_folder_path = lightcurve.astrometry_path
    with open(lightcurve.astrometry_path.joinpath("models.pickle"), 'rb') as f:
        models = pickle.load(f)

    dp = models['dp']
    tp2px_model = models['tp2px']
    ref2tp_model = models['ref2tp']
    ref2px_model = models['ref2px']

    reference_exposure = lightcurve.get_reference_exposure()
    reference_index = dp.exposure_map[reference_exposure]
    reference_mask = (dp.exposure_index == reference_index)

    # Gaia magnitude distribution
    plt.subplots(nrows=1, ncols=1, figsize=(6., 6.))
    plt.hist(list(set(dp.cat_mag)), bins='auto', histtype='step', color='black')
    plt.grid()
    plt.savefig(lightcurve.astrometry_path.joinpath("gaia_mag_dist.png"), dpi=200.)

    ################################################################################
    ################################################################################
    ################################################################################
    # Control plots for ref2px model
    #

    logger.info("Plotting control plots for the ref2px model")

    # First get stars that exist in both reference exposure and other exposures
    gaiaid_in_ref = dp.gaiaid[dp.exposure_index == reference_index]
    measure_mask = np.where(dp.exposure_index != reference_index, np.isin(dp.gaiaid, gaiaid_in_ref), False)
    in_ref = np.hstack([np.where(gaiaid == gaiaid_in_ref) for gaiaid in dp.gaiaid[measure_mask]]).flatten()

    ref2px_residuals = ref2px_model.residuals(np.array([dp.x[dp.exposure_index==reference_index][in_ref], dp.y[dp.exposure_index==reference_index][in_ref]]),
                                              np.array([dp.x[measure_mask], dp.y[measure_mask]]),
                                              space_indices=dp.exposure_index[measure_mask])

    ref2px_save_folder_path= save_folder_path.joinpath("ref2px_plots")

    ################################################################################
    # Residuals scatter plot
    lims = (np.min(ref2px_residuals), np.max(ref2px_residuals))

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10., 10.), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [4, 1]})
    plt.suptitle("Ref->px residuals scatter")
    ax = plt.subplot(2, 2, 1)
    plt.plot(ref2px_residuals[0], ref2px_residuals[1], ',')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("$x$ [pixel]")
    plt.ylabel("$y$ [pixel]")
    plt.grid()

    ax = plt.subplot(2, 2, 2)
    x = np.linspace(*lims, 500)
    m, s = norm.fit(ref2px_residuals[1])
    plt.hist(ref2px_residuals[1], histtype='step', orientation='horizontal', density=True, bins='auto', color='black')
    plt.plot(norm.pdf(x, loc=m, scale=s), x, color='black')
    plt.text(0.1, 0.8, "$\sigma={0:.4f}$".format(s), transform=ax.transAxes, fontsize=13)
    plt.text(0.1, 0.77, "$\mu={0:.4f}$".format(m), transform=ax.transAxes, fontsize=13)
    plt.ylim(lims)
    plt.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.subplot(2, 2, 3)
    m, s = norm.fit(ref2px_residuals[0])
    plt.plot(x, norm.pdf(x, loc=m, scale=s), color='black')
    plt.hist(ref2px_residuals[0], histtype='step', density=True, bins='auto', color='black')
    plt.text(0.75, 0.82, "$\sigma={0:.4f}$".format(s), transform=ax.transAxes, fontsize=13)
    plt.text(0.75, 0.7, "$\mu={0:.4f}$".format(m), transform=ax.transAxes, fontsize=13)
    plt.xlim(lims)
    plt.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_scatter.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals distributions
    plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.hist(ref2px_residuals[0], bins=100, histtype='step', color='black')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.hist(ref2px_residuals[1], bins=100, histtype='step', color='black')
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_dist.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / magnitude
    plt.subplots(ncols=1, nrows=2, figsize=(15., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(dp.mag[measure_mask], ref2px_residuals[0]/np.sqrt(dp.sx[measure_mask]**2+dp.sy[measure_mask]**2), ',', color='black')
    # plt.ylim([-0.2, 0.2])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(dp.mag[measure_mask], ref2px_residuals[1]/np.sqrt(dp.sx[measure_mask]**2+dp.sy[measure_mask]**2), ',', color='black')
    # plt.ylim([-0.2, 0.2])
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_mag.png"), dpi=300.)
    plt.close()

    ################################################################################
    # Residuals binplot / magnitude
    plt.subplots(nrows=2, ncols=2, figsize=(18., 10.))
    plt.subplot(2, 2, 1)
    #xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag[measure_mask], ref2px_residuals[0]/np.sqrt(dp.sx[measure_mask]**2+dp.sy[measure_mask]**2), nbins=5, data=True, rms=True, scale=False)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag[measure_mask], ref2px_residuals[0], nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag[measure_mask], ref2px_residuals[1], nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{git}}$ [pixel]")
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_binplot_mag.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / magnitude
    plt.subplots(ncols=1, nrows=2, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(dp.centered_color[measure_mask], ref2px_residuals[0], ',', color='black')
    plt.ylim([-0.2, 0.2])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(dp.centered_color[measure_mask], ref2px_residuals[1], ',', color='black')
    plt.ylim([-0.2, 0.2])
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_color.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals binplot / color
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.centered_color[measure_mask], ref2px_residuals[0], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.centered_color[measure_mask], ref2px_residuals[1], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{fit}}$ [pixel]")
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_binplot_color.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Partial chi2 per exposure/gaia star
    ref2px_chi2_exposure = np.bincount(dp.exposure_index[measure_mask], weights=np.sqrt(ref2px_residuals[0]**2+ref2px_residuals[1]**2))/np.bincount(dp.exposure_index[measure_mask])
    ref2px_chi2_gaiaid = np.bincount(dp.gaiaid_index[measure_mask], weights=np.sqrt(ref2px_residuals[0]**2+ref2px_residuals[1]**2))/np.bincount(dp.gaiaid_index[measure_mask])
    ref2px_chi2_gaiaid = np.pad(ref2px_chi2_gaiaid, (0, len(dp.gaiaid_set) - len(ref2px_chi2_gaiaid)), constant_values=np.nan)

    df_ref2px_chi2_exposure = pd.DataFrame(data=ref2px_chi2_exposure, index=dp.exposure_set, columns=['chi2'])
    df_ref2px_chi2_exposure.to_csv(ref2px_save_folder_path.joinpath("chi2_exposures.csv"), sep=",")

    coeffs_df = pd.DataFrame(data=ref2px_model.coeffs, index=dp.exposure_set)
    coeffs_df.to_csv(ref2px_save_folder_path.joinpath("ref2px_coeffs.csv"), sep=",")

    plt.figure()
    plt.plot(range(len(ref2px_chi2_exposure)), ref2px_chi2_exposure, '.')
    plt.xlabel("Exposure index")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("chi2_exposure.png"), dpi=200.)
    plt.close()

    plt.figure()
    plt.plot(range(len(ref2px_chi2_gaiaid)), ref2px_chi2_gaiaid, '.')
    plt.xlabel("Star index")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("chi2_gaiaid.png"), dpi=200.)
    plt.close()
    ################################################################################

    ################################################################################
    # Athmospheric refraction / residuals
    plt.subplots(ncols=1, nrows=2, figsize=(20., 10.))
    plt.suptitle("Ref -> px - Athmospheric refraction / residuals")
    plt.subplot(2, 1, 1)
    plt.plot(np.tan(np.deg2rad(dp.z[measure_mask]))*dp.parallactic_angle_x[measure_mask]*dp.centered_color[measure_mask], ref2px_residuals[0], ',')
    # idx2marker = {0: '*', 1: '.', 2: 'o', 3: 'x'}
    # for i, rcid in enumerate(model.dp.rcid_set):
    #     rcid_mask = (model.dp.rcid == rcid)
    #     plt.scatter(np.tan(np.deg2rad(model.dp.z[rcid_mask]))*model.dp.parallactic_angle_x[rcid_mask][:, 0]*(model.dp.color[rcid_mask]-color_mean), res_x[rcid_mask], marker=idx2marker[i], label=rcid, s=0.1)

    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\sin(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$x-x_\\mathrm{fit}$")
    # plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.tan(np.deg2rad(dp.z[measure_mask]))*dp.parallactic_angle_y[measure_mask]*dp.centered_color[measure_mask], ref2px_residuals[1], ',')
    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\cos(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$y-y_\\mathrm{fit}$")
    plt.grid()

    plt.savefig(ref2px_save_folder_path.joinpath("atmref_residuals.pdf"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / distance to origin
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(dp.x[measure_mask]**2+dp.y[measure_mask]**2), ref2px_residuals[0], ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$x-x_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(dp.x[measure_mask]**2+dp.y[measure_mask]**2), ref2px_residuals[1], ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$y-y_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.savefig(ref2px_save_folder_path.joinpath("residuals_origindistance.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Star chi2 / proper motion
    pm = [3.6e6*365.25*np.sqrt(dp.pmdec[dp.gaiaid_index==gaiaid_index][0]**2+dp.pmra[dp.gaiaid_index==gaiaid_index][0]**2*np.cos(dp.dec[dp.gaiaid_index==gaiaid_index][0])**2) for gaiaid_index in range(len(dp.gaiaid_set))]

    plt.figure()
    plt.suptitle("Proper motion distribution")
    plt.hist(pm, bins='auto', histtype='step', color='black')
    plt.grid()
    plt.xlabel("$\\sqrt{\\mu_{\\alpha^2\\ast}+\\mu_\\delta^2}$ [mas/yr]")
    plt.ylabel("Count")
    plt.xlim(0., max(pm))
    plt.savefig(save_folder_path.joinpath("proper_motion_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residual vectors / exposure
    ref2px_save_folder_path.joinpath("scatter").mkdir(exist_ok=True)
    for exposure in dp.exposure_set:
        exposure_index = dp.exposure_map[exposure]
        plt.subplots(nrows=1, ncols=2, figsize=(11., 5.))
        plt.suptitle("Residual scatter/vector plot for {}".format(exposure))
        exposure_mask = (dp.exposure_index[measure_mask] == exposure_index)

        plt.subplot(1, 2, 1)
        plt.quiver(dp.x[measure_mask][exposure_mask], dp.y[measure_mask][exposure_mask], ref2px_residuals[0][exposure_mask], ref2px_residuals[1][exposure_mask])
        plt.xlim(0., quadrant_width_px)
        plt.ylim(0., quadrant_height_px)
        plt.xlabel("$x$ [pixel]")
        plt.ylabel("$y$ [pixel]")

        plt.subplot(1, 2, 2)
        plt.plot(ref2px_residuals[0][exposure_mask], ref2px_residuals[1][exposure_mask], ".", color='black')
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.6, 0.6)
        # plt.axis('equal')
        plt.grid()
        plt.xlabel("$x$ [pixel]")

        plt.savefig(ref2px_save_folder_path.joinpath("scatter/{}_residuals_vector.png".format(exposure)))
        plt.close()
    ################################################################################

    ################################################################################
    ################################################################################
    ################################################################################
    # Control plots for ref2tp model
    #
    logger.info("Plotting control plots for the ref2tp model")

    ref2tp_residuals = ref2tp_model.residuals(np.array([dp.x[reference_mask], dp.y[reference_mask]]),
                                              np.array([dp.tpx[reference_mask], dp.tpy[reference_mask]]))

    ref2tp_save_folder_path= save_folder_path.joinpath("ref2tp_plots")
    ref2tp_save_folder_path.mkdir(exist_ok=True)

    plt.figure()
    plt.suptitle("Ref -> tp residuals scatter (on reference exposure)")
    plt.plot(ref2tp_residuals[0], ref2tp_residuals[1], '.')
    plt.grid()
    plt.axis('equal')
    plt.savefig(ref2tp_save_folder_path.joinpath("residuals_scatter.png"), dpi=300.)
    plt.close()

    ################################################################################
    ################################################################################
    ################################################################################
    # Residual distribution for tp2px model
    #
    logger.info("Plotting control plots for the tp2px model")

    tp2px_residuals = tp2px_model.residuals(np.array([dp.tpx, dp.tpy]), np.array([dp.x, dp.y]), np.array([dp.pmtpx, dp.pmtpy]), dp.mjd, exposure_indices=dp.exposure_index)

    tp2px_save_folder_path = save_folder_path.joinpath("tp2px_plots")
    tp2px_save_folder_path.mkdir(exist_ok=True)

    ################################################################################
    # Residuals scatter plot

    lims = (np.min(tp2px_residuals), np.max(tp2px_residuals))

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10., 10.), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [4, 1]})
    plt.suptitle("Tp->px residuals scatter")
    ax = plt.subplot(2, 2, 1)
    plt.plot(tp2px_residuals[0], tp2px_residuals[1], ',')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("$x$ [pixel]")
    plt.ylabel("$y$ [pixel]")
    plt.grid()

    ax = plt.subplot(2, 2, 2)
    x = np.linspace(*lims)
    m, s = norm.fit(tp2px_residuals[1])
    plt.hist(tp2px_residuals[1], histtype='step', orientation='horizontal', density=True, bins='auto', color='black')
    plt.plot(norm.pdf(x, loc=m, scale=s), x, color='black')
    plt.text(0.1, 0.8, "$\sigma={0:.4f}$".format(s), transform=ax.transAxes, fontsize=13)
    plt.text(0.1, 0.77, "$\mu={0:.4f}$".format(m), transform=ax.transAxes, fontsize=13)
    plt.ylim(lims)
    plt.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.subplot(2, 2, 3)
    m, s = norm.fit(tp2px_residuals[0])
    plt.plot(x, norm.pdf(x, loc=m, scale=s), color='black')
    plt.hist(tp2px_residuals[0], histtype='step', density=True, bins='auto', color='black')
    plt.text(0.75, 0.82, "$\sigma={0:.4f}$".format(s), transform=ax.transAxes, fontsize=13)
    plt.text(0.75, 0.7, "$\mu={0:.4f}$".format(m), transform=ax.transAxes, fontsize=13)
    plt.xlim(lims)
    plt.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.savefig(tp2px_save_folder_path.joinpath("residuals_scatter.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals distribution
    plt.subplot(1, 2, 1)
    plt.hist(tp2px_residuals[0], bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(1, 2, 2)
    plt.hist(tp2px_residuals[1], bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(tp2px_save_folder_path.joinpath("residuals_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Magnitude / residuals
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(dp.mag, tp2px_residuals[0], ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(dp.mag, tp2px_residuals[1], ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(tp2px_save_folder_path.joinpath("magnitude_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Magnitude / residuals binplot
    plt.subplots(nrows=2, ncols=2, figsize=(10., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, tp2px_residuals[0], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, tp2px_residuals[1], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{git}}$ [pixel]")

    plt.savefig(tp2px_save_folder_path.joinpath("magnitude_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color plot
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(dp.centered_color, tp2px_residuals[0], ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(dp.centered_color, tp2px_residuals[1], ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(tp2px_save_folder_path.joinpath("color_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color binplot
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.centered_color, tp2px_residuals[0], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.centered_color, tp2px_residuals[1], nbins=5, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.savefig(tp2px_save_folder_path.joinpath("color_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Partial chi2 per exposure/gaia star
    tp2px_chi2_exposure = np.bincount(dp.exposure_index, weights=np.sqrt(tp2px_residuals[0]**2+tp2px_residuals[1]**2))/np.bincount(dp.exposure_index)
    tp2px_chi2_gaiaid = np.bincount(dp.gaiaid_index, weights=np.sqrt(tp2px_residuals[0]**2+tp2px_residuals[1]**2))/np.bincount(dp.gaiaid_index)
    df_tp2px_chi2_exposure = pd.DataFrame(data=tp2px_chi2_exposure, index=dp.exposure_set, columns=['chi2'])
    df_tp2px_chi2_exposure.to_csv(tp2px_save_folder_path.joinpath("chi2_exposures.csv"), sep=",")

    plt.figure()
    plt.plot(range(len(tp2px_chi2_exposure)), tp2px_chi2_exposure, '.')
    plt.xlabel("Exposure index")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(tp2px_save_folder_path.joinpath("chi2_exposure.png"), dpi=200.)
    plt.close()

    plt.figure()
    plt.plot(range(len(tp2px_chi2_gaiaid)), tp2px_chi2_gaiaid, '.')
    plt.xlabel("Star index")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(tp2px_save_folder_path.joinpath("chi2_gaiaid.png"), dpi=200.)
    plt.close()
    ################################################################################


    ################################################################################
    ################################################################################
    # Parallactic angle distribution
    plt.subplot(1, 2, 1)
    plt.hist(dp.parallactic_angle_x, bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(dp.parallactic_angle_y, bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("Count")

    plt.savefig(save_folder_path.joinpath("parallactic_angle_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residual vectors / exposure
    tp2px_save_folder_path.joinpath("scatter").mkdir(exist_ok=True)
    for exposure in dp.exposure_set:
        exposure_index = dp.exposure_map[exposure]
        plt.subplots(nrows=1, ncols=2, figsize=(11., 5.))
        plt.suptitle("Residual scatter/vector plot for {}".format(exposure))
        exposure_mask = (dp.exposure_index == exposure_index)

        plt.subplot(1, 2, 1)
        plt.quiver(dp.x[exposure_mask], dp.y[exposure_mask], tp2px_residuals[0][exposure_mask], tp2px_residuals[1][exposure_mask])
        plt.xlim(0., quadrant_width_px)
        plt.ylim(0., quadrant_height_px)
        plt.xlabel("$x$ [pixel]")
        plt.ylabel("$y$ [pixel]")

        plt.subplot(1, 2, 2)
        plt.plot(tp2px_residuals[0][exposure_mask], tp2px_residuals[1][exposure_mask], ".", color='black')
        plt.axis('equal')
        plt.grid()
        plt.xlabel("$x$ [pixel]")
        # plt.ylabel("$y$ [pixel]")

        plt.savefig(tp2px_save_folder_path.joinpath("scatter/{}_residuals_vector.png".format(exposure)))
        plt.close()
    ################################################################################
    ################################################################################
    # Residuals / exposure
    # save_folder_path.joinpath("parallactic_angle_exposure").mkdir(exist_ok=True)
    # for exposure in model.dp.exposure_set:
    #     exposure_mask = (model.dp.exposure == exposure)
    #     plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
    #     plt.subplot(1, 2, 1)
    #     plt.quiver(model.dp.x[exposure_mask], model.dp.y[exposure_mask], model.dp.parallactic_angle_x[exposure_mask], model.dp.parallactic_angle_y[quadrant_mask])
    #     plt.xlim(0., utils.exposure_width_px)
    #     plt.ylim(0., utils.exposure_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.subplot(1, 2, 2)
    #     plt.quiver(model.dp.x[exposure_mask], model.dp.y[exposure_mask], res_x[exposure_mask], res_y[quadrant_mask])
    #     plt.xlim(0., utils.exposure_width_px)
    #     plt.ylim(0., utils.exposure_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.savefig(save_folder_path.joinpath("parallactic_angle_exposure/parallactic_angle_{}.png".format(exposure)), dpi=150.)
    #     plt.close()

    ################################################################################
    # # Color distribution
    plt.hist(dp.centered_color, bins=25, histtype='step', color='black')
    plt.xlabel("$B_p-R_p-\\left<B_p-R_p\\right>$ [mag]")
    plt.ylabel("Count")
    plt.grid()

    plt.savefig(save_folder_path.joinpath("color_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Athmospheric refraction / residuals
    plt.subplots(ncols=1, nrows=2, figsize=(20., 10.))
    plt.subplot(2, 1, 1)
    plt.plot(np.tan(np.deg2rad(dp.z))*dp.parallactic_angle_x*dp.centered_color, tp2px_residuals[0], ',')
    # idx2marker = {0: '*', 1: '.', 2: 'o', 3: 'x'}
    # for i, rcid in enumerate(model.dp.rcid_set):
    #     rcid_mask = (model.dp.rcid == rcid)
    #     plt.scatter(np.tan(np.deg2rad(model.dp.z[rcid_mask]))*model.dp.parallactic_angle_x[rcid_mask][:, 0]*(model.dp.color[rcid_mask]-color_mean), res_x[rcid_mask], marker=idx2marker[i], label=rcid, s=0.1)

    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\sin(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$x-x_\\mathrm{fit}$")
    #plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.tan(np.deg2rad(dp.z))*dp.parallactic_angle_y*dp.centered_color, tp2px_residuals[1], ',')
    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\cos(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$y-y_\\mathrm{fit}$")
    plt.grid()

    plt.savefig(tp2px_save_folder_path.joinpath("atmref_residuals.pdf"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2/exposure / seeing
    plt.plot(dp.seeing, tp2px_chi2_exposure[dp.exposure_index], '.')
    plt.xlabel("Seeing")
    plt.ylabel("$\\chi^2_\\mathrm{exposure}$")
    plt.grid()

    plt.savefig(tp2px_save_folder_path.joinpath("chi2_exposure_seeing.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2/exposure / airmass
    plt.plot(dp.airmass, tp2px_chi2_exposure[dp.exposure_index], '.')
    plt.xlabel("Airmass")
    plt.ylabel("$\\chi^2_\\mathrm{exposure}$")
    plt.grid()

    plt.savefig(tp2px_save_folder_path.joinpath("chi2_exposure_airmass.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / distance to origin
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(dp.x**2+dp.y**2), tp2px_residuals[0], ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$x-x_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(dp.x**2+dp.y**2), tp2px_residuals[1], ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$y-y_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.savefig(tp2px_save_folder_path.joinpath("residuals_origindistance.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / star index
    plt.plot(range(len(dp.gaiaid_set)), tp2px_chi2_gaiaid, ".", color='black')
    plt.xlabel("Gaia #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(tp2px_save_folder_path.joinpath("chi2_star.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / exposure index
    plt.plot(range(len(dp.exposure_set)), tp2px_chi2_exposure, ".", color='black')
    plt.xlabel("Exposure #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(tp2px_save_folder_path.joinpath("chi2_exposure.png"), dpi=300.)
    plt.close()
    ################################################################################
