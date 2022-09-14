#!/usr/bin/env python3

import time

import imageproc.composable_functions as compfuncs
import saunerie.fitparameters as fp
from scipy import sparse
import numpy as np
from sksparse import cholmod

from utils import get_wcs_from_quadrant, quadrant_width_px, quadrant_height_px

def wcs_residuals(band_path, ztfname, filtercode, logger, args):
    """

    """

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from saunerie.plottools import binplot
    matplotlib.use('Agg')

    save_folder = band_path.joinpath("wcs_residuals_plots")
    save_folder.mkdir(exist_ok=True)
    matched_stars_df = pd.read_parquet(band_path.joinpath("matched_stars.parquet"))

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

    plt.savefig(save_folder.joinpath("wcs_res_dist.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/magnitude
    plt.subplots(nrows=2, ncols=1, figsize=(15., 10.))
    plt.subplot(2, 1, 1)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['x']-matched_stars_df['gaia_x'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmde']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.scatter(matched_stars_df['mag'], matched_stars_df['y']-matched_stars_df['gaia_y'], c=np.sqrt(matched_stars_df['pmra']**2+matched_stars_df['pmde']**2), marker='+', s=0.05)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{Gaia}$ [pixel]")
    plt.colorbar()
    plt.grid()

    plt.savefig(save_folder.joinpath("mag_wcs_res.png"), dpi=750.)
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

    plt.savefig(save_folder.joinpath("mag_wcs_res_binplot.png"), dpi=300.)
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
    plt.savefig(save_folder.joinpath("rms_mean_lc.png"), dpi=300.)
    plt.close()
    ################################################################################


class AstromModel():
    """

    """
    def __init__(self, dp, degree=5, scale_quadrant=True, quadrant_size=(3072, 3080)):
        self.dp = dp
        self.degree = degree
        self.params = self.init_params()
        self.params['k'].fix()

        self.quadrant_size = quadrant_size
        self.scale = (1./quadrant_size[0], 1./quadrant_size[1])

    def init_params(self):
        self.tp_to_pix = compfuncs.BiPol2D(deg=self.degree, key='quadrant', n=len(self.dp.quadrant_set))
        return fp.FitParameters([*self.tp_to_pix.get_struct(), ('k', 1)])

    @property
    def sigma(self):
        return np.hstack((self.dp.sx, self.dp.sy))

    @property
    def W(self):
        return sparse.dia_array((1./self.sigma**2, 0), shape=(2*len(self.dp.nt), 2*len(self.dp.nt)))

    def __call__(self, x, p=None, jac=False):
        if p is not None:
            self.params.free = p

        k = self.params['k'].full
        centered_color = self.dp.color - np.mean(self.dp.color)

        if not jac:
            xy = self.tp_to_pix((self.dp.tpx, self.dp.tpy), p=self.params, quadrant=self.dp.quadrant_index)

            # Could be better implemented
            # xy[0] = xy[0] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_x*centered_color
            # xy[1] = xy[1] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_y*centered_color

            return xy
        else:
            # Derivatives wrt polynomial
            xy, _, (i, j, vals) = self.tp_to_pix.derivatives(np.array([self.dp.tpx, self.dp.tpy]),
                                                              p=self.params, quadrant=self.dp.quadrant_index)

            # Could be better implemented
            # xy[0] = xy[0] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_x*centered_color
            # xy[1] = xy[1] + k*np.tan(np.deg2rad(self.dp.z))*self.dp.parallactic_angle_y*centered_color

            ii = [np.hstack([i, i+len(self.dp.nt)])]
            jj = [np.tile(j, 2).ravel()]
            vv = [np.hstack(vals).ravel()]

            # dm/dk
            i = np.arange(2*len(self.dp.nt))
            ii.append(i)
            jj.append(np.full(2*len(self.dp.nt), self.params['k'].indexof(0)))
            vv.append(np.tile(np.tan(np.deg2rad(self.dp.z)), 2)*np.concatenate([self.dp.parallactic_angle_x, self.dp.parallactic_angle_y])*np.tile(centered_color, 2))

            NN = 2*len(self.dp.nt)
            ii = np.hstack(ii)
            jj = np.hstack(jj)
            vv = np.hstack(vv)
            ok = jj >= 0
            J_model = sparse.coo_array((vv[ok], (ii[ok], jj[ok])), shape=(NN, len(self.params.free)))

            return xy, J_model

    def residuals(self):
        fit_x, fit_y = self.__call__(None, self.params.free)
        return self.dp.x - fit_x, self.dp.y - fit_y


def _fit_astrometry(model, logger):
    logger.info("Astrometry fit with {} measurements.".format(len(model.dp.nt)))
    t = time.perf_counter()
    p = model.params.free.copy()
    v, J = model(None, p, jac=1)
    H = J.T @ model.W @ J
    B = J.T @ model.W @ np.hstack((model.dp.x, model.dp.y))
    fact = cholmod.cholesky(H.tocsc())
    p = fact(B)
    model.params.free = p
    logger.info("Done. Elapsed time={}.".format(time.perf_counter()-t))
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


def astrometry_fit(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    import numpy as np
    from imageproc import gnomonic
    import matplotlib
    import matplotlib.pyplot as plt
    from croaks import DataProxy
    from utils import get_ref_quadrant_from_band_folder, ztf_latitude, BiPol2D_fit, create_2D_mesh_grid
    matplotlib.use('Agg')

    reference_quadrant = get_ref_quadrant_from_band_folder(band_path)

    sn_parameters_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='sn_info')

    # Define plot saving folder
    save_folder = band_path.joinpath("astrometry_plots")
    save_folder.mkdir(exist_ok=True)

    # Load data
    matched_stars_df = pd.read_parquet(band_path.joinpath("matched_stars.parquet"))
    matched_stars_df = matched_stars_df.loc[matched_stars_df['quadrant']==reference_quadrant]

    # Compute parallactic angle
    parallactic_angle_sin = np.cos(np.deg2rad(ztf_latitude))*np.sin(np.deg2rad(matched_stars_df['ha']))/np.sin(np.deg2rad(matched_stars_df['z']))
    parallactic_angle_cos = np.sqrt(1.-parallactic_angle_sin**2)

    # Project to tangent plane
    tpx, tpy, e_tpx, e_tpy = gnomonic.gnomonic_projection(np.deg2rad(matched_stars_df['ra'].to_numpy()), np.deg2rad(matched_stars_df['dec'].to_numpy()),
                                                          np.deg2rad(sn_parameters_df['sn_ra'].to_numpy()), np.deg2rad(sn_parameters_df['sn_dec'].to_numpy()),
                                                          np.zeros_like(matched_stars_df['ra'].to_numpy()), np.zeros_like(matched_stars_df['dec'].to_numpy()))

    # Add paralactic angle
    matched_stars_df['parallactic_angle_x'] = parallactic_angle_sin
    matched_stars_df['parallactic_angle_y'] = parallactic_angle_cos

    matched_stars_df['tpx'] = tpx[0]
    matched_stars_df['tpy'] = tpy[0]

    plt.plot(tpx[0], tpy[0], '.')
    plt.axis('equal')
    plt.savefig(save_folder.joinpath("tangent_plane_positions.png"), dpi=300.)
    plt.close()

    matched_stars_df['color'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']

    # Do cut in magnitude
    matched_stars_df = matched_stars_df.loc[matched_stars_df['mag'] < -10.]

    # Build dataproxy for model
    dp = DataProxy(matched_stars_df.to_records(),
                   x='x', sx='sx', sy='sy', y='y', ra='ra', dec='dec', quadrant='quadrant', mag='mag', gaiaid='gaiaid',
                   bpmag='bpmag', rpmag='rpmag', seeing='seeing', z='z', airmass='airmass', tpx='tpx', tpy='tpy',
                   parallactic_angle_x='parallactic_angle_x', parallactic_angle_y='parallactic_angle_y', color='color', rcid='rcid')

    dp.make_index('quadrant')
    dp.make_index('gaiaid')
    dp.make_index('color')
    dp.make_index('rcid')


    ################################################################################
    # Tangent space to pixel space

    # Build model
    tp2px_model = AstromModel(dp, degree=3)
    tp2px_model.init_params()

    # Model fitting
    _fit_astrometry(tp2px_model, logger)
    res_x, res_y = tp2px_model.residuals()

    # Filter outlier quadrants
    model = _filter_noisy(tp2px_model, res_x, res_y, 'quadrant', 0.1, logger)

    # Redo fit
    _fit_astrometry(tp2px_model, logger)
    res_x, res_y = tp2px_model.residuals()

    print("k={}".format(tp2px_model.params['k'].full.item()))


    ################################################################################
    # Pixel space to tangent space for reference quadrant
    #

    print("Computing reference pixel space to tangent space transformation...")
    grid_res = 30
    def _ref2tp_polymodel(degree=3):
        print("Reference quadrant={}".format(reference_quadrant))
        wcs = get_wcs_from_quadrant(band_path.joinpath(reference_quadrant))

        # project corner points to sky then to tangent plane
        print("  Projecting corner points to tangent plane using WCS")
        corner_points_px = np.array([[0., 0.],
                                    [quadrant_width_px, 0.],
                                    [0., quadrant_height_px],
                                    [quadrant_width_px, quadrant_height_px]])
        print(corner_points_px)
        corner_points_radec = wcs.pixel_to_world_values(corner_points_px)
        print(corner_points_radec)
        corner_points_radec = np.vstack(corner_points_radec).T
        print(corner_points_radec)
        [corner_points_tpx], [corner_points_tpy], _, _ = gnomonic.gnomonic_projection(np.deg2rad(corner_points_radec[0]), np.deg2rad(corner_points_radec[1]),
                                                                                      np.deg2rad(sn_parameters_df['sn_ra'].to_numpy()), np.deg2rad(sn_parameters_df['sn_dec'].to_numpy()),
                                                                                      np.zeros(4), np.zeros(4))

        plt.plot(corner_points_tpx, corner_points_tpy, "o")
        print("====")
        print(np.min(corner_points_tpx), np.max(corner_points_tpx))
        print(np.min(corner_points_tpy), np.max(corner_points_tpy))
        print("  Creating 2D mesh of resolution {}x{}".format(grid_res, grid_res))
        grid_points_px = create_2D_mesh_grid(np.linspace(0., quadrant_width_px, grid_res), np.linspace(0., quadrant_height_px, grid_res))

        grid_points_tp = create_2D_mesh_grid(np.linspace(np.min(corner_points_tpx), np.max(corner_points_tpx), grid_res),
                                             np.linspace(np.min(corner_points_tpy), np.max(corner_points_tpy), grid_res))
        # grid_points_tp = create_2D_mesh_grid(np.linspace(np.max(corner_points_tpx), np.min(corner_points_tpx), grid_res),
        #                                      np.linspace(np.max(corner_points_tpy), np.min(corner_points_tpy), grid_res))

        plt.plot(grid_points_tp[:, 0], grid_points_tp[:, 1], 'x')
        print(grid_points_tp)
        print("  Fitting using polynomial of degree {}".format(degree))
        band_path.joinpath("astrometry_plots/ref2tp_plots").mkdir(exist_ok=True)
        ref2tp_model = BiPol2D_fit(grid_points_px.T, grid_points_tp.T, degree, control_plots=band_path.joinpath("astrometry_plots/ref2tp_plots"))
        totp = ref2tp_model(grid_points_px.T)
        plt.plot(totp[0], totp[1], '.')
        plt.savefig("out0.png", dpi=1000.)
        plt.close()
        return ref2tp_model

    ref2tp_model = _ref2tp_polymodel(degree=3)

    print("Done")

    ################################################################################
    # Reference pixel space to pixel space

    print("Transforming reference pixel space to quadrants pixel space")

    # Point grid position in quadrant (pixel) space
    grid_points_px = create_2D_mesh_grid(np.linspace(0., quadrant_width_px, grid_res), np.linspace(0., quadrant_height_px, grid_res))

    # Reference pixel space to tangent space
    ref_grid_tp = ref2tp_model(grid_points_px.T)

    # Reference tangent space to quadrant pixel space for each quadrant
    space_indices = np.concatenate([np.full(ref_grid_tp.shape[1], i) for i, _ in enumerate(dp.quadrant_set)])

    ref_grid_px = tp2px_model.tp_to_pix(np.tile(ref_grid_tp, (1, len(dp.quadrant_set))), tp2px_model.params, quadrant=space_indices)

    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(ref_grid_tp[0], ref_grid_tp[1], '.')
    plt.plot(ref_grid_tp[0][:2], ref_grid_tp[1][:2], 'x')
    plt.subplot(1, 2, 2)
    plt.plot(ref_grid_px[0][0], ref_grid_px[1][0], '.')
    plt.plot(grid_points_px[:, 0][0], grid_points_px[:, 1][0], 'x')
    plt.savefig("out2.png", dpi=1000.)
    plt.close()
    print("Done")

    ref_idx = dp.quadrant_map[reference_quadrant]
    ref_mask = (space_indices==ref_idx)
    #print(ref_grid_px[:, ref_mask])

    print("Composing polynomials to get ref -> quadrant in pixel space")
    # Polynomials for reference pixel space to quadrant pixel spaces
    band_path.joinpath("astrometry_plots/ref2px").mkdir(exist_ok=True)
    ref2px_model = BiPol2D_fit(np.tile(grid_points_px.T, (1, len(dp.quadrant_set))), ref_grid_px, 3, space_indices, control_plots=band_path.joinpath("astrometry_plots/ref2px"))

    print("Done")

    # coeffs_dict = dict((key, ref2px_model.coeffs[key]) for key in ref2px_model.coeffs.keys())
    coeffs_df = pd.DataFrame(data=ref2px_model.coeffs, index=model.dp.quadrant_set)
    print(coeffs_df)
    coeffs_df.to_csv(save_folder.joinpath("coeffs.csv"), sep=",")

    print("Ref -> ref residuals")
    ref_mask = (dp.quadrant_index==ref_idx)
    print(sum(ref_mask))
    print(np.vstack([dp.x, dp.y])[:, ref_mask].shape)
    print(ref2px_model.residuals(np.vstack([dp.x, dp.y])[:, ref_mask], np.vstack([dp.x, dp.y])[:, ref_mask], space_index=dp.quadrant_index[ref_mask]))
    print(ref_idx)

    return


def astrometry_fit_init():
    pass


def astrometry_fit_plot(cwd, ztfname, filtercode, logger, args):
    from sksparse import cholmod
    from imageproc import gnomonic
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    from saunerie.plottools import binplot
    import imageproc.composable_functions as compfuncs
    import saunerie.fitparameters as fp
    from scipy import sparse
    from croaks import DataProxy

    ################################################################################
    # Control plots

    # Extract and save on disk polynomial coefficients
    coeffs_dict = dict((key, model.params[key].full) for key in model.params._struct.slices.keys())
    coeffs_df = pd.DataFrame(data=coeffs_dict, index=model.dp.quadrant_set)
    coeffs_df.to_csv(save_folder.joinpath("coeffs.csv"), sep=",")

    # Compute partial Chi2 per quadrant and per gaia star
    chi2_quadrant = np.bincount(model.dp.quadrant_index, weights=np.sqrt(res_x**2+res_y**2))/np.bincount(model.dp.quadrant_index)
    chi2_gaiaid = np.bincount(model.dp.gaiaid_index, weights=np.sqrt(res_x**2+res_y**2))/np.bincount(model.dp.gaiaid_index)

    color_mean = np.mean(model.dp.color_set)

    ################################################################################
    # Parallactic angle distribution
    plt.subplot(1, 2, 1)
    plt.hist(matched_stars_df['parallactic_angle_x'], bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("#")


    plt.hist(matched_stars_df['parallactic_angle_y'], bins=100)
    plt.grid()
    plt.xlabel("$\\sin(\eta)$")
    plt.ylabel("#")

    plt.savefig(save_folder.joinpath("parallactic_angle_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / quadrant
    # save_folder.joinpath("parallactic_angle_quadrant").mkdir(exist_ok=True)
    # for quadrant in model.dp.quadrant_set:
    #     quadrant_mask = (model.dp.quadrant == quadrant)
    #     plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
    #     plt.subplot(1, 2, 1)
    #     plt.quiver(model.dp.x[quadrant_mask], model.dp.y[quadrant_mask], model.dp.parallactic_angle_x[quadrant_mask], model.dp.parallactic_angle_y[quadrant_mask])
    #     plt.xlim(0., utils.quadrant_width_px)
    #     plt.ylim(0., utils.quadrant_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.subplot(1, 2, 2)
    #     plt.quiver(model.dp.x[quadrant_mask], model.dp.y[quadrant_mask], res_x[quadrant_mask], res_y[quadrant_mask])
    #     plt.xlim(0., utils.quadrant_width_px)
    #     plt.ylim(0., utils.quadrant_height_px)
    #     plt.xlabel("$x$ [pixel]")
    #     plt.xlabel("$y$ [pixel]")

    #     plt.savefig(save_folder.joinpath("parallactic_angle_quadrant/parallactic_angle_{}.png".format(quadrant)), dpi=150.)
    #     plt.close()

    ################################################################################
    # Color distribution
    plt.hist(model.dp.color_set-color_mean, bins=25)
    plt.xlabel("$B_p-R_p-\\left<B_p-R_p\\right>$ [mag]")

    plt.savefig(save_folder.joinpath("color_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Athmospheric refraction / residuals
    plt.subplots(ncols=1, nrows=2, figsize=(20., 10.))
    plt.subplot(2, 1, 1)
    plt.plot(np.tan(np.deg2rad(model.dp.z))*model.dp.parallactic_angle_x*(model.dp.color-color_mean), res_x, ',')
    # idx2marker = {0: '*', 1: '.', 2: 'o', 3: 'x'}
    # for i, rcid in enumerate(model.dp.rcid_set):
    #     rcid_mask = (model.dp.rcid == rcid)
    #     plt.scatter(np.tan(np.deg2rad(model.dp.z[rcid_mask]))*model.dp.parallactic_angle_x[rcid_mask][:, 0]*(model.dp.color[rcid_mask]-color_mean), res_x[rcid_mask], marker=idx2marker[i], label=rcid, s=0.1)

    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\sin(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$x-x_\\mathrm{fit}$")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(np.tan(np.deg2rad(model.dp.z))*model.dp.parallactic_angle_y*(model.dp.color-color_mean), res_y, ',')
    plt.ylim(-0.5, 0.5)
    plt.xlabel("$\\tan(z)\\cos(\\eta)(B_p-R_p-\\left<B_p-R_p\\right>)$")
    plt.ylabel("$y-y_\\mathrm{fit}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("atmref_residuals.pdf"), dpi=300.)
    plt.close()

    ################################################################################
    # Chi2/quadrant / seeing
    plt.plot(model.dp.seeing, chi2_quadrant[model.dp.quadrant_index], '.')
    plt.xlabel("Seeing")
    plt.ylabel("$\\chi^2_\\mathrm{quadrant}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("chi2_quadrant_seeing.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2/quadrant / airmass
    plt.plot(model.dp.airmass, chi2_quadrant[model.dp.quadrant_index], '.')
    plt.xlabel("Airmass")
    plt.ylabel("$\\chi^2_\\mathrm{quadrant}$")
    plt.grid()

    plt.savefig(save_folder.joinpath("chi2_quadrant_airmass.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals / distance to origin
    plt.subplots(nrows=1, ncols=2, figsize=(10., 5.))
    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(model.dp.x**2+model.dp.y**2), res_x, ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$x-x_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(model.dp.x**2+model.dp.y**2), res_y, ',')
    plt.xlabel("$D(x,y)$ [pixel]")
    plt.ylabel("$y-y_\\mathrm{model}$ [pixel]")
    plt.grid()
    plt.savefig(save_folder.joinpath("residuals_origindistance.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / star index
    plt.plot(range(len(model.dp.gaiaid_set)), chi2_gaiaid, ".", color='black')
    plt.xlabel("Gaia #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(save_folder.joinpath("chi2_star.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Chi2 / quadrant index
    plt.plot(range(len(model.dp.quadrant_set)), chi2_quadrant, ".", color='black')
    plt.xlabel("Quadrant #")
    plt.ylabel("$\\chi^2$")
    plt.grid()
    plt.savefig(save_folder.joinpath("chi2_quadrant.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals distribution
    plt.subplot(1, 2, 1)
    plt.hist(res_x, bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(1, 2, 2)
    plt.hist(res_y, bins=100, range=[-0.25, 0.25])
    plt.grid()
    plt.xlabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("residuals_distribution.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Magnitude / residuals
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(model.dp.mag, res_x, ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(model.dp.mag, res_y, ",")
    plt.grid()
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("magnitude_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    plt.subplots(nrows=2, ncols=2, figsize=(10., 10.))
    # Magnitude / residuals binplot
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, res_x, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(dp.mag, res_y, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$m$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{git}}$ [pixel]")

    plt.savefig(save_folder.joinpath("magnitude_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color plot
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.subplot(2, 1, 1)
    plt.plot(model.dp.bpmag-model.dp.rpmag, res_x, ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")

    plt.subplot(2, 1, 2)
    plt.plot(model.dp.bpmag-model.dp.rpmag, res_y, ",")
    plt.grid()
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")

    plt.savefig(save_folder.joinpath("color_residuals.png"), dpi=300.)
    plt.close()
    ################################################################################

    ################################################################################
    # Residuals/color binplot
    plt.subplots(nrows=2, ncols=2, figsize=(20., 10.))
    plt.subplot(2, 2, 1)
    xbinned_mag, yplot_res, res_dispersion = binplot(model.dp.bpmag-model.dp.rpmag, res_x, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$x-x_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{x-x_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 3)
    xbinned_mag, yplot_res, res_dispersion = binplot(model.dp.bpmag-model.dp.rpmag, res_y, nbins=10, data=True, rms=True, scale=False)
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$y-y_\\mathrm{fit}$ [pixel]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xbinned_mag, res_dispersion, color='black')
    plt.xlabel("$B_p-R_p$ [mag]")
    plt.ylabel("$\\sigma_{y-y_\\mathrm{fit}}$ [pixel]")
    plt.grid()

    plt.savefig(save_folder.joinpath("color_residuals_binplot.png"), dpi=300.)
    plt.close()
    ################################################################################

