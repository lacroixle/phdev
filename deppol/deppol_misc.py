#!/usr/bin/env python3


def psf_study(exposure, logger, args):
    from utils import ListTable, match_pixel_space, quadrant_size_px
    import numpy as np
    from saunerie.plottools import binplot
    from scipy.stats import norm
    import matplotlib
    import matplotlib.pyplot as plt
    import pickle
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    from numpy.polynomial.polynomial import Polynomial
    # matplotlib.use('Agg')

    print(exposure.name)
    if not exposure.path.joinpath("psfstars.list").exists():
        return True

    # psf_resid = ListTable.from_filename(quadrant_path.joinpath("psf_resid_tuple.dat"))
    psf_stars_df = exposure.get_matched_catalog('psfstars')
    aper_stars_df = exposure.get_matched_catalog('aperstars')
    gaia_stars_df = exposure.get_matched_ext_catalog('gaia')

    def _aperflux_to_mag(aper, cat_df):
        cat_df['mag_{}'.format(aper)] = -2.5*np.log10(cat_df[aper])
        cat_df['emag_{}'.format(aper)] = 1.08*cat_df['e{}'.format(aper)]/cat_df[aper]

    aper_stars_df['mag'] = -2.5*np.log10(aper_stars_df['flux'])
    aper_stars_df['emag'] = 1.08*aper_stars_df['eflux']/aper_stars_df['flux']
    psf_stars_df['mag'] = -2.5*np.log10(psf_stars_df['flux'])
    psf_stars_df['emag'] = 1.08*np.log10(psf_stars_df['eflux'])
    _aperflux_to_mag('apfl0', aper_stars_df)
    _aperflux_to_mag('apfl4', aper_stars_df)
    _aperflux_to_mag('apfl6', aper_stars_df)
    _aperflux_to_mag('apfl8', aper_stars_df)

    d_mag = (psf_stars_df['mag'] - aper_stars_df['mag_apfl6']).to_numpy()
    d_mag = d_mag - np.nanmean(d_mag)
    ed_mag = np.sqrt(psf_stars_df['emag']**2+aper_stars_df['emag_apfl6']**2).to_numpy()

    d_mag_mask = ~np.isnan(d_mag)
    d_mag = d_mag[d_mag_mask]
    ed_mag = ed_mag[d_mag_mask]

    min_mag, max_mag = 13., 21.
    low_bin, high_bin = 14, 17

    gmag_binned, d_mag_binned, ed_mag_binned = binplot(gaia_stars_df.iloc[d_mag_mask]['Gmag'].to_numpy(), d_mag, robust=True, data=False, scale=True, weights=1./ed_mag**2, bins=np.linspace(min_mag, max_mag, 15), color='black', zorder=15)

    bin_mask = (ed_mag_binned > 0.)
    gmag_binned = gmag_binned[bin_mask]
    d_mag_binned = np.array(d_mag_binned)[bin_mask]
    ed_mag_binned = ed_mag_binned[bin_mask]

    poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 0, w=1./ed_mag_binned, full=True)
    poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 1, w=1./ed_mag_binned, full=True)
    poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 2, w=1./ed_mag_binned, full=True)

    poly0_chi2 = poly0_chi2/(len(gmag_binned)-1)
    poly1_chi2 = poly1_chi2/(len(gmag_binned)-2)
    poly2_chi2 = poly2_chi2/(len(gmag_binned)-3)

    # plt.subplots(ncols=1, nrows=2, figsize=(10., 6.), sharex=True)
    # plt.suptitle("Skewness of stars as a function of magnitude")
    # plt.subplot(2, 1, 1)
    # plt.plot(gaia_stars_df['Gmag'].to_numpy(), aper_stars_df['gmx3'].to_numpy(), '.')
    # plt.grid()
    # plt.axhline(0.)
    # plt.ylabel("$M_x^3$")
    # plt.subplot(2, 1, 2)
    # plt.plot(gaia_stars_df['Gmag'].to_numpy(), aper_stars_df['gmy3'].to_numpy(), '.')
    # plt.grid()
    # plt.axhline()
    # plt.ylabel("$M_y^3$")
    # plt.xlabel("$m_\\mathrm{Gaia}$ [mag]")
    # plt.show()

    mag_space = np.linspace(min_mag, max_mag, 100)

    plt.figure()
    plt.subplots(ncols=1, nrows=1, figsize=(10., 6.))
    plt.suptitle("{}\n{}\nSky level={}ADU".format(exposure.name, poly0_chi2-poly2_chi2, exposure.exposure_header['sexsky']))
    plt.plot(gaia_stars_df['Gmag'].to_numpy(), d_mag, '.')
    plt.plot(mag_space, poly0(mag_space))
    plt.plot(mag_space, poly2(mag_space))
    plt.ylabel("$m_\\mathrm{Aper}-m_\\mathrm{PSF}$ [mag]")
    plt.xlabel("$m_\\mathrm{Gaia}$ [mag]")
    plt.xlim(13., 21.)
    plt.ylim(-2., 2.)
    plt.grid()
    plt.show()

    with open(exposure.path.joinpath("psfskewness.pickle"), 'wb') as f:
        pickle.dump(f, {'exposure': exposure.name,
                        'poly0_chi2': poly0_chi2,
                        'poly1_chi2': poly1_chi2,
                        'poly2_chi2': poly2_chi2,
                        'skylev': exposure.exposure_header['sexsky']})

    return True
    # psf_residuals_px = (psf_resid.df['fimg'] - psf_resid.df['fpsf']).to_numpy()

    subx = 4
    suby = 1

    # Fit order 2 polynomials on skewness/mag relation
    def _fit_skewness_subquadrants(axis, sub):
        ranges = np.linspace(0., quadrant_size_px[axis], sub+1)
        skew_polynomials = []
        for i in range(sub):
            range_mask = (stand_stars_df['y'] >= ranges[i]) & (stand_stars_df['y'] < ranges[i+1])
            skew_polynomials.append(np.polynomial.Polynomial.fit(stand_stars_df.loc[range_mask]['mag'], stand_stars_df.loc[range_mask]['gm{}3'.format(axis)], 1))

        return skew_polynomials

    skew_polynomial_x = np.polynomial.Polynomial.fit(stand_stars_df['mag'], stand_stars_df['gmx3'], 1)
    skew_polynomial_y = np.polynomial.Polynomial.fit(stand_stars_df['mag'], stand_stars_df['gmy3'], 1)

    skew_polynomial_subx = _fit_skewness_subquadrants('x', subx)
    skew_polynomial_suby = _fit_skewness_subquadrants('y', suby)

    with open(exposure.path.joinpath("stamp_skewness.pickle"), 'wb') as f:
        pickle.dump({'poly_x': skew_polynomial_x, 'poly_y': skew_polynomial_y,
                     'poly_x_sub': skew_polynomial_subx, 'poly_y_sub': skew_polynomial_suby}, f)

    plt.figure(figsize=(7., 7.))
    ax = plt.axes()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.suptitle("Skewness plane for standalone stars stamps in \n {}".format(quadrant_path.name))
    plt.scatter(stand_stars_df['gmx3'], stand_stars_df['gmy3'], s=2., color='black')
    plt.xlabel("$M^g_{xxx}$")
    plt.ylabel("$M^g_{yyy}$")
    plt.axvline(0.)
    plt.axhline(0.)
    plt.axvline(skew_polynomial_x.coef[0], ls='--')
    plt.axhline(skew_polynomial_y.coef[0], ls='--')
    plt.axis('equal')
    plt.savefig(quadrant_path.joinpath("{}_mx3_my3_plane.png".format(quadrant_path.name)), dpi=300.)
    plt.close()

    plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(10., 5.))
    plt.suptitle("Skewness vs magnitude for standalone stars stamps in {}".format(quadrant_path.name))

    ax = plt.subplot(2, 1, 1)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.scatter(stand_stars_df['mag'], stand_stars_df['gmx3'], s=2.)
    # plt.plot(mag, skew_polynomial_x(mag), color='black')
    plt.axhline(0.)
    # plt.ylim(-0.5, 0.3)
    # plt.xlim(-15., -7)
    plt.ylabel("$M^g_{xxx}$")

    ax = plt.subplot(2, 1, 2)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.scatter(stand_stars_df['mag'], stand_stars_df['gmy3'], s=2.)
    # plt.plot(mag, skew_polynomial_y(mag), color='black')
    plt.axhline(0.)
    plt.xlabel("$m$")
    # plt.ylim(-0.3, 0.3)
    # plt.xlim(-15., -6)
    plt.ylabel("$M^g_{yyy}$")

    plt.savefig(exposure.path.joinpath("{}_skewness_magnitude.png".format(quadrant_path.name)), dpi=300.)
    plt.close()
    return

    # star_indices = match_pixel_space(psf_stars_df[['x', 'y']].to_records(), psf_resid_df[['xc', 'yc']].rename(mapper={'xc': 'x', 'yc': 'y'}, axis='columns').to_records())
    # psf_size = int(np.sqrt(sum(star_indices==0)))
    # psf_residuals = np.zeros([len(psf_stars.df), psf_size*psf_size])
    # for star_index in range(len(np.bincount(star_indices))):
    #     star_mask = (star_indices==star_index)
    #     ij = (psf_resid.df.iloc[star_mask]['j']*psf_size + psf_resid.df.iloc[star_mask]['i'] + int(np.floor(psf_size**2/2))).to_numpy()
    #     np.put_along_axis(psf_residuals[star_index], ij, psf_residuals_px[star_mask], 0)

    # psf_residuals = psf_residuals.reshape(len(psf_stars.df), psf_size, psf_size)

    # bins_count = 5
    # mag_range = (psf_stars.df['mag'].min(), psf_stars.df['mag'].max())
    # bins = np.linspace(*mag_range, bins_count+1)
    # psf_residual_means = []
    # psf_residual_count = []
    # for i in range(bins_count):
    #     lower_bound = bins[i]
    #     upper_bound = bins[i+1]

    #     binned_stars_mask = np.all([psf_stars.df['mag'] < upper_bound, psf_stars.df['mag'] >= lower_bound], axis=0)
    #     psf_residual_means.append(np.mean(psf_residuals[binned_stars_mask], axis=0))
    #     psf_residual_count.append(sum(binned_stars_mask))

    # plt.subplots(ncols=bins_count, nrows=1, figsize=(5.*bins_count, 5.))
    # plt.suptitle("Binned PSF residuals for {}".format(quadrant_path.name))
    # for i, psf_residual_mean in enumerate(psf_residual_means):
    #     plt.subplot(1, bins_count,  1+i)
    #     plt.imshow(psf_residual_mean)
    #     plt.axis('off')
    #     plt.title("${0:.2f} \leq m < {1:.2f}$\n$N={2}$".format(bins[i], bins[i+1], psf_residual_count[i]))

    # plt.savefig(quadrant_path.joinpath("{}_psf_residuals.png".format(quadrant_path.name)))
    # plt.close()
    # plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
    # # plt.subplot(1, 2, 1)
    # # plt.scatter(stand.df['gmx3'], stand.df['gmy3'], c=stand.df['x'], s=2.)
    # # plt.axvline(0.)
    # # plt.axhline(0.)

    # # plt.subplot(1, 2, 2)
    # # plt.scatter(stand.df['gmx3'], stand.df['gmy3'], c=stand.df['y'], s=2.)
    # # plt.axvline(0.)
    # # plt.axhline(0.)
    # # plt.show()

    # plt.subplot(2, 2, 1)
    # plt.scatter(stand.df['mag'], stand.df['gmx3'], c=stand.df['x'], s=1.)
    # plt.axhline(0.)
    # plt.subplot(2, 2, 2)
    # plt.scatter(stand.df['mag'], stand.df['gmy3'], c=stand.df['x'])
    # plt.axhline(0.)
    # plt.subplot(2, 2, 3)
    # plt.scatter(stand.df['mag'], stand.df['gmx3'], c=stand.df['y'])
    # plt.axhline(0.)
    # plt.subplot(2, 2, 4)
    # plt.scatter(stand.df['mag'], stand.df['gmy3'], c=stand.df['y'])
    # plt.axhline(0.)
    # plt.show()

    # psf_resid.df.to_csv("psf_resid_tuple.csv", sep=",")
    # psf_tuple.df.to_csv("psftuple.csv", sep=",")
    # psf_stars.df.to_csv("psfstars.csv", sep=",")


    # psf_resid.df['flux'] = psf_stars.df['flux'].iloc[idx].reset_index(drop=True)
    # psf_resid.df['eflux'] = psf_stars.df['eflux'].iloc[idx].reset_index(drop=True)

    # limits = [np.min(res), np.max(res)]
    # f = np.linspace(*limits, 200)
    # s = np.var(res)
    # m = np.mean(res)
    # print(np.sqrt(s))
    # print(m)

    # plt.hist(res, bins=1000, density=True)
    # plt.plot(f, norm.pdf(f, loc=m, scale=np.sqrt(s)))
    # plt.xlim(limits)
    # plt.show()

    # plt.plot(psf_resid.df['flux'], psf_resid.df['eflux']/psf_resid.df['flux'], '.')
    # plt.show()

    # plt.subplots(nrows=2, ncols=1, figsize=(10., 5.), sharex=True)
    # plt.subplot(2, 1, 1)
    # binned_mag, plot_res, res_dispersion = binplot(psf_resid.df['flux'].to_numpy(), psf_resid.df['fimg'] - psf_resid.df['fpsf'], nbins=10, data=True, rms=True, scale=False)

    # plt.ylabel("res")
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(binned_mag, res_dispersion, color='black')
    # plt.xlabel("Flux [ADU]")
    # plt.ylabel("$\sigma_\\mathrm{res}$")
    # plt.grid()

    # plt.show()

    return True


def psf_study_reduce(band_path, ztfname, filtercode, logger, args):
    import numpy as np
    import pandas as pd
    from deppol_utils import quadrants_from_band_path
    from utils import get_header_from_quadrant_path, ListTable
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    from utils import idx2markerstyle, plot_ztf_focal_plan_values
    import itertools
    import pickle
    from astropy import time
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    # matplotlib.use('Agg')

    output_path = band_path.joinpath("psf_study")
    output_path.mkdir(exist_ok=True)

    # Compare seeing computed by sextractor vs the one computed by the ZTF pipeline
    quadrant_paths = quadrants_from_band_path(band_path, logger, check_files="psfstars.list", ignore_noprocess=False)

    #chi2_df = pd.read_csv(band_path.joinpath("astrometry/ref2px_plots/chi2_quadrants.csv"), index_col=0)

    # def _extract_seeing(quadrant_path):
    #     hdr = get_header_from_quadrant_path(quadrant_path)
    #     if quadrant_path.name in chi2_df.index.tolist():
    #         chi2 = chi2_df.at[quadrant_path.name, 'chi2']
    #     else:
    #         chi2 = float('nan')

    #     psfstars_list = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))
    #     return 2.355*hdr['gfseeing'], hdr['seeing'], len(psfstars_list.df), chi2

    # seeings = np.array(list(map(_extract_seeing, quadrant_paths))).T

    def _extract_skewness(quadrant_path):
        hdr = get_header_from_quadrant_path(quadrant_path)
        with open(quadrant_path.joinpath("stamp_skewness.pickle"), 'rb') as f:
            poly = pickle.load(f)
            poly_x = poly['poly_x']
            poly_y = poly['poly_y']
            poly_x_sub = poly['poly_x_sub']
            poly_y_sub = poly['poly_y_sub']

        #out_dict = {'mjd': hdr['obsmjd'], 'quadrant_id': "{}_{}".format(hdr['ccdid'], hdr['qid'])}
        out_dict = {'mjd': hdr['obsmjd'], 'ccdid': int(hdr['ccdid']), 'qid': int(hdr['qid']), 'rcid': int(hdr['rcid']),
                    'sky': hdr['sexsky'], 'skysigma': hdr['sexsigma'], 'gfseeing': hdr['gfseeing'], 'seeing': hdr['seeing']}

        out_dict.update(dict(('x{}'.format(i), coef) for i, coef in enumerate(poly_x.coef)))
        out_dict.update(dict(('y{}'.format(i), coef) for i, coef in enumerate(poly_y.coef)))
        out_dict.update({'x1_sub': [poly.coef[1] for poly in poly_x_sub],
                         'y1_sub': [poly.coef[1] for poly in poly_y_sub]})

        return out_dict

    skewness_df = pd.DataFrame(list(map(_extract_skewness, quadrant_paths)))
    cmap = 'coolwarm'

    unique_mjds = list(set(skewness_df['mjd']))
    xskewness = dict([(mjd, dict([(i+1, dict([(j, None) for j in range(4)])) for i in range(16)])) for mjd in unique_mjds])
    yskewness = dict([(mjd, dict([(i+1, dict([(j, None) for j in range(4)])) for i in range(16)])) for mjd in unique_mjds])

    for mjd in unique_mjds:
        skewness_today_df = skewness_df.loc[skewness_df['mjd']==mjd]
        for row in skewness_today_df.iterrows():
            # Dirty
            row = row[1]
            ccdid = int(row['ccdid'])
            qid = int(row['qid'])

            xskewness[mjd][ccdid][qid-1] = np.tile(np.array(row['x1_sub']), (len(row['x1_sub']), 1))
            yskewness[mjd][ccdid][qid-1] = np.tile(np.array(row['y1_sub']), (len(row['y1_sub']), 1))

    def _plot_focal_plane_skewness(x, vmin, vmax):
        cm = ScalarMappable(cmap=cmap)
        cm.set_clim(vmin=vmin, vmax=vmax)

        if x == 'x1':
            skewness = xskewness
        elif x == 'y1':
            skewness = yskewness

        for mjd in unique_mjds:
            t = time.Time(mjd, format='mjd')
            fig = plt.figure(figsize=(5., 6.), constrained_layout=True)
            f1, f2 = fig.subfigures(ncols=1, nrows=2, height_ratios=[8., 1.])
            plot_ztf_focal_plan_values(f1, skewness[mjd], vmin=vmin, vmax=vmax, cmap=cmap)
            ax = f2.add_subplot()
            Colorbar(ax, cm, orientation='horizontal', label="${}_{}$".format(x[0], x[1]))
            fig.suptitle("Focal plane skewness in ${}$ direction the {}".format(x[0], t.to_value('iso', subfmt='date')), fontsize='large')
            plt.savefig(output_path.joinpath("{}_focal_plane_{}skewness.png".format(t.to_value('iso', subfmt='date'), x[0])), dpi=300.)
            plt.close()

    sharedscale = False
    center = True
    if sharedscale:
        vmin, vmax = np.min([skewness_df['x1'], skewness_df['y1']]), np.max([skewness_df['x1'], skewness_df['y1']])
        xvmin = vmin
        xvmax = vmax
        yvmin = vmin
        yvmax = vmax
    else:
        xvmin, xvmax = np.min(skewness_df['x1']), np.max(skewness_df['x1'])
        yvmin, yvmax = np.min(skewness_df['y1']), np.max(skewness_df['y1'])

    if center:
        if sharedscale:
            m = max(np.abs(vmin), np.abs(vmax))
            xvmin = -m
            xvmax = m
            yvmin = -m
            yvmax = m
        else:
            m = max(np.abs(xvmin), np.abs(xvmax))
            xvmin = -m
            xvmax = m
            m = max(np.abs(yvmin), np.abs(yvmax))
            yvmin = -m
            yvmax = m

    _plot_focal_plane_skewness('x1', xvmin, xvmax)
    _plot_focal_plane_skewness('y1', yvmin, yvmax)

    rcids = set(skewness_df['rcid'])
    plt.subplots(nrows=2, ncols=1, figsize=(10., 5.))
    plt.suptitle("Order 1 polynomial coefficients fit on skewness/mag")
    ax = plt.subplot(2, 1, 1)
    if len(rcids) <= len(idx2markerstyle):
        for i, rcid in enumerate(rcids):
            rcid_mask = (skewness_df['rcid'] == rcid)
            plt.plot(skewness_df.loc[rcid_mask]['mjd'], skewness_df[rcid_mask]['x1'], idx2markerstyle[i], color='black', label=rcid)
        plt.legend(title="Quadrant ID")
    else:
        plt.scatter(skewness_df['mjd'], skewness_df['x1'], c=skewness_df['rcid'].tolist(), s=1.)
        plt.colorbar()
    plt.axhline(0., color='black')
    plt.ylim(vmin, vmax)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.xlabel("MJD")
    plt.ylabel("$S^x_1$")

    ax = plt.subplot(2, 1, 2)
    if len(rcids) <= len(idx2markerstyle):
        for i, rcid in enumerate(rcids):
            rcid_mask = (skewness_df['rcid'] == rcid)
            plt.plot(skewness_df.loc[rcid_mask]['mjd'], skewness_df[rcid_mask]['y1'], idx2markerstyle[i], color='black', label=rcid)
        plt.legend(title="Quadrant ID")
    else:
        plt.scatter(skewness_df['mjd'], skewness_df['y1'], c=skewness_df['rcid'].tolist(), s=1.)
        plt.colorbar()
    plt.axhline(0., color='black')
    plt.ylim(vmin, vmax)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.xlabel("MJD")
    plt.ylabel("$S^y_1$")

    plt.tight_layout()
    plt.savefig(output_path.joinpath("skewness_mjd.png"), dpi=300.)
    plt.close()

    # ax = plt.subplot(2, 1, 2)
    # if len(qids) <= len(idx2markerstyle):
    #     for i, qid in enumerate(qids):
    #         qid_mask = (skewness_df['quadrant_id'] == qid)
    #         plt.plot(skewness_df.loc[qid_mask]['mjd'], skewness_df[qid_mask]['y1'], idx2markerstyle[i], color='black', label=qid)
    #     plt.legend(title="Quadrant ID")
    # else:
    #     plt.scatter(skewness_df['mjd'], skewness_df['y1'], c=skewness_df['quadrant_id'].tolist(), s=1.)
    #     plt.colorbar()
    # plt.axhline(0., color='black')
    # plt.ylim(*ylim)
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.xlabel("MJD")
    # plt.ylabel("$S^y_1$")

    return

    # ax = plt.axes()
    # plt.plot(seeings[0], seeings[1], '.')
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.xlabel("GF seeing [pixel]")
    # plt.ylabel("ZTF seeing [pixel]")
    # plt.show()
    # plt.close()

    # ax = plt.axes()
    # plt.plot(seeings[2], seeings[3], '.')
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.xlabel("PSF stars count")
    # plt.ylabel("Chi2")
    # plt.show()
    # plt.close()

    # ax = plt.axes()
    # plt.plot(seeings[2], seeings[0]-seeings[1]-np.mean(seeings[0]-seeings[1]), '.')
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.xlabel("PSF stars count")
    # plt.ylabel("GF - ZTF - <GF-ZTF> (seeing) [pixel]")
    # plt.show()
    # plt.close()

    # ax = plt.axes()
    # plt.plot(seeings[3], seeings[0]-seeings[1]-np.mean(seeings[0]-seeings[1]), '.')
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.xlabel("Chi2")
    # plt.ylabel("GF - ZTF - <GF-ZTF> (seeing) [pixel]")
    # plt.show()
    # plt.close()

    # ax = plt.axes()
    # plt.hist(seeings[0]-seeings[1]-np.mean(seeings[0]-seeings[1]), bins='auto', histtype='step', color='black')
    # ax.tick_params(which='both', direction='in')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # plt.grid(linestyle='--', color='xkcd:sky blue')
    # plt.show()


# def seeing_study(band_path, ztfname, filtercode, logger, args):
#     from utils import get_header_from_quadrant_path, plot_ztf_focal_plan_values, ListTable
#     from deppol_utils import quadrants_from_band_path
#     import pickle
#     import numpy as np
#     import itertools
#     import matplotlib
#     import matplotlib.pyplot as plt
#     from matplotlib.cm import ScalarMappable
#     from matplotlib.colorbar import Colorbar
#     quadrant_paths = quadrants_from_band_path(band_path, logger, check_files="psfstars.list", ignore_noprocess=False)

#     keys = {'seeing': 'hdr',
#             'seseeing': 'hdr',
#             'gfseeing': 'gfseeing',
#             'momentx1': 'psfstars',
#             'momenty1': 'psfstars',
#             'momentx2': 'psfstars',
#             'momenty2': 'psfstars',
#             'momentxy': 'psfstars'}

#     if not band_path.joinpath("seeings.pickle").exists() or args.from_scratch:
#         vals = {}

#         for key in keys:
#             vals[key] = {}
#             for i in range(1, 17):
#                 vals[key][i] = {}
#                 for j in range(4):
#                     vals[key][i][j] = []

#         for quadrant_path in quadrant_paths:
#             hdr = get_header_from_quadrant_path(quadrant_path)
#             psfstars_list = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))

#             for key in keys.keys():
#                 if keys[key] == 'psfstars':
#                     vals[key][hdr['ccdid']][hdr['qid']-1].append(psfstars_list.header[key])
#                 elif key == 'seseeing':
#                     vals[key][hdr['ccdid']][hdr['qid']-1].append(2.355*hdr[key])
#                 elif keys[key] == 'hdr':
#                     vals[key][hdr['ccdid']][hdr['qid']-1].append(hdr[key])
#                 elif key == 'gfseeing':
#                     with open(quadrant_path.joinpath("psf.dat"), 'r') as f:
#                         lines = f.readlines()
#                         gfseeing = float(lines[3].split(" ")[0].strip())
#                     vals[key][hdr['ccdid']][hdr['qid']-1].append(2.355*gfseeing)

#         with open(band_path.joinpath("seeings.pickle"), 'wb') as f:
#             pickle.dump(vals, f)
#     else:
#         with open(band_path.joinpath("seeings.pickle"), 'rb') as f:
#             vals = pickle.load(f)

#     for key in keys:
#         for key_i in vals[key].keys():
#             for key_j in vals[key][key_i].keys():
#                 vals[key][key_i][key_j] = np.median(list(map(lambda x: float(x), vals[key][key_i][key_j])))

#     cmap = 'coolwarm'
#     def _plot_focal_plane_seeing(key):
#         val_list = [[vals[key][key_i][key_j] for key_j in vals[key][key_i].keys()] for key_i in vals[key].keys()]
#         val_list = list(itertools.chain(*val_list))
#         vmin = min(val_list)
#         vmax = max(val_list)

#         cm = ScalarMappable(cmap=cmap)
#         cm.set_clim(vmin=vmin, vmax=vmax)

#         fig = plt.figure(figsize=(5., 6.), constrained_layout=False)
#         #f1, f2, f3 = fig.subfigures(ncols=1, nrows=3, height_ratios=[12., 1., 1.])
#         f1, f2 = fig.subfigures(ncols=1, nrows=2, height_ratios=[12., 0.3], wspace=-1., hspace=-1.)
#         f1.suptitle("\n{}-{} Focal plane {}".format(ztfname, filtercode, key), fontsize='large')
#         plot_ztf_focal_plan_values(f1, vals[key], vmin=vmin, vmax=vmax, cmap=cmap, scalar=True)
#         ax2 = f2.add_subplot(clip_on=False)
#         f2.subplots_adjust(bottom=-5, top=5.)
#         ax2.set_position((0.2, 0, 0.6, 1))
#         # ax2 = f2.add_subplot()
#         # ax3 = f3.add_subplot(axis='none')
#         # Colorbar(ax, cm, orientation='horizontal')
#         cb = f1.colorbar(cm, cax=ax2, orientation='horizontal')
#         # ax2.autoscale(tight=True)
#         ax2.set_clip_on(False)
#         ax2.minorticks_on()
#         ax2.tick_params(direction='inout')
#         plt.savefig(band_path.joinpath("{}-{}_focal_plane_{}.png".format(ztfname, filtercode, key)), dpi=300., bbox_inches='tight', pad_inches=0.1)
#         plt.close()

#     for key in keys:
#         # print("Plotting {}".format(key))
#         _plot_focal_plane_seeing(key)


def retrieve_catalogs(lightcurve, logger, args):
    from ztfquery.fields import get_rcid_centroid
    from ztfimg.catalog import download_vizier_catalog
    import pandas as pd
    import numpy as np
    from croaks.match import NearestNeighAssoc


    # Retrieve Gaia and PS1 catalogs
    exposures = lightcurve.get_exposures()
    field_rcid_pairs = list(set([(exposure.field, exposure.rcid) for exposure in exposures]))
    centroids = [get_rcid_centroid(rcid, field) for field, rcid in field_rcid_pairs]

    logger.info("Retrieving catalogs for (fieldid, rcid) pairs: {}".format(field_rcid_pairs))

    name_to_objid = {'gaia': 'Source', 'ps1': 'objID'}

    def _download_catalog(name, centroids):
        catalogs = [download_vizier_catalog(name, centroid, radius=0.6) for centroid in centroids]
        #catalog_df = pd.concat(catalogs).set_index(name_to_objid[name]).drop_duplicates()
        catalog_df = pd.concat(catalogs).drop_duplicates(ignore_index=True, subset=name_to_objid[name])
        return catalog_df

    def _get_catalog(name, centroids):
        catalog_path = lightcurve.ext_catalogs_path.joinpath("{}_full.parquet".format(name))
        if not catalog_path.exists():
            logger.info("Downloading catalog {}... ".format(name))
            catalog_df = _download_catalog(name, centroids)
            catalog_path.parents[0].mkdir(exist_ok=True)

            # Remove low detection counts PS1 objects
            if name == 'ps1':
                catalog_df = catalog_df.loc[catalog_df['Nd']>=30].reset_index(drop=True)

            catalog_df.to_parquet(catalog_path)
            logger.info("Saving catalog into {}".format(catalog_path))
        else:
            catalog_df = pd.read_parquet(catalog_path)

        return catalog_df


    logger.info("Retrieving external catalogs")
    gaia_df = _get_catalog('gaia', centroids)
    ps1_df = _get_catalog('ps1', centroids)

    logger.info("Matching external catalogs")
    assoc = NearestNeighAssoc(first=[gaia_df['ra'].to_numpy(), gaia_df['dec'].to_numpy()], radius = 2./60./60.)
    i = assoc.match(ps1_df['ra'].to_numpy(), ps1_df['dec'].to_numpy())

    gaia_df = gaia_df.iloc[i[i>=0]].reset_index(drop=True)
    ps1_df = ps1_df.iloc[i>=0].reset_index(drop=True)

    logger.info("Saving matched catalogs")
    # Changing some units
    gaia_df['pmRA'] = gaia_df['pmRA']/np.cos(np.deg2rad(gaia_df['dec']))/1000./3600./365.25
    gaia_df['pmDE'] = gaia_df['pmDE']/1000./3600./365.25
    gaia_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("gaia.parquet"))
    ps1_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("ps1.parquet"))

    return True


def match_catalogs(exposure, logger, args):
    from itertools import chain
    from utils import contained_in_exposure, match_pixel_space, gaiarefmjd, quadrant_width_px, quadrant_height_px
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    try:
        gaia_stars_df = exposure.lightcurve.get_ext_catalog('gaia')
    except FileNotFoundError as e:
        logger.error("Could not find Gaia catalog!")
        return False

    if not exposure.func_status('makepsf'):
        logger.error("makepsf was not successful, no psfstars catalog!")
        return False

    psf_stars_df = exposure.get_catalog("psfstars.list").df
    aper_stars_df = exposure.get_catalog("standalone_stars.list").df

    wcs = exposure.wcs
    mjd = exposure.mjd

    exposure_centroid = exposure.get_centroid()
    exposure_centroid_skycoord = SkyCoord(ra=exposure_centroid[0], dec=exposure_centroid[1], unit='deg')

    # Match aperture photometry and psf photometry catalogs in pixel space
    i = match_pixel_space(psf_stars_df[['x', 'y']], aper_stars_df[['x', 'y']], radius=2.)

    psf_indices = i[i>=0]
    aper_indices = np.arange(len(aper_stars_df))[i>=0]

    psf_stars_df = psf_stars_df.iloc[psf_indices].reset_index(drop=True)
    aper_stars_df = aper_stars_df.iloc[aper_indices].reset_index(drop=True)

    logger.info("Matched {} PSF/aper stars".format(len(psf_stars_df)))
    # Match Gaia catalog with exposure catalogs
    # First remove Gaia stars not contained in the exposure
    gaia_stars_skycoords = SkyCoord(ra=gaia_stars_df['ra'].to_numpy(), dec=gaia_stars_df['dec'].to_numpy(), unit='deg')

    # sep = exposure_centroid_skycoord.separation(gaia_stars_skycoords)
    # idxc = (sep < 0.6*u.deg)
    # gaia_stars_skycoords = gaia_stars_skycoords[idxc]
    # gaia_stars_df = gaia_stars_df[idxc]

    # Project Gaia stars into pixel space
    gaia_stars_x, gaia_stars_y = gaia_stars_skycoords.to_pixel(wcs)
    gaia_stars_df['x'] = gaia_stars_x
    gaia_stars_df['y'] = gaia_stars_y

    # Match exposure catalogs to Gaia stars
    try:
        i = match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), psf_stars_df[['x', 'y']].to_records(), radius=2.)
    except Exception as e:
        logger.error("Could not match any Gaia stars!")
        raise ValueError("Could not match any Gaia stars!")

    gaia_indices = i[i>=0]
    cat_indices = np.arange(len(psf_indices))[i>=0]

    # Save indices into an HDF file
    with pd.HDFStore(exposure.path.joinpath("cat_indices.hd5"), 'w') as hdfstore:
        hdfstore.put('psfstars_indices', pd.Series(psf_indices))
        hdfstore.put('aperstars_indices', pd.Series(aper_indices))
        hdfstore.put('ext_cat_indices', pd.DataFrame({'x': gaia_stars_x[gaia_indices], 'y': gaia_stars_y[gaia_indices], 'indices': gaia_indices}))
        hdfstore.put('cat_indices', pd.Series(cat_indices))


    matched_gaia_stars_df = gaia_stars_df.iloc[gaia_indices].reset_index(drop=True)

    logger.info("Matched {} GAIA stars".format(len(matched_gaia_stars_df)))

    return True


def clean(exposure, logger, args):
    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["elixir.fits", "dead.fits.gz", ".dbstuff"]
    files_to_delete = list(filter(lambda f: f.name not in files_to_keep, list(exposure.path.glob("*"))))

    for file_to_delete in files_to_delete:
            file_to_delete.unlink()

    return True


def clean_reduce(lightcurve, logger, args):
    from shutil import rmtree

    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["prepare.log"]

    # Delete all files
    files_to_delete = list(filter(lambda f: f.is_file() and (f.name not in files_to_keep), list(lightcurve.path.glob("*"))))

    [f.unlink() for f in files_to_delete]

    # Delete output folders
    # We delete everything but the exposure folders
    # folders_to_keep = list(band_path.glob("ztf_*"))
    folders_to_keep = [exposure.path for exposure in lightcurve.get_exposures(ignore_noprocess=True)]
    folders_to_delete = [folder for folder in list(lightcurve.path.glob("*")) if (folder not in folders_to_keep) or folder.is_file()]

    #[rmtree(band_path.joinpath(folder), ignore_errors=True) for folder in folders_to_delete]
    [rmtree(folder, ignore_errors=True) for folder in folders_to_delete]


def filter_psfstars_count(lightcurve, logger, args):
    from utils import ListTable

    exposures = lightcurve.get_exposures(files_to_check="makepsf.success")
    flagged = []

    if not args.min_psfstars:
        logger.info("min-psfstars not defined. Computing it from astro-degree.")
        logger.info("astro-degree={}".format(args.astro_degree))
        min_psfstars = (args.astro_degree+1)*(args.astro_degree+2)
        logger.info("Minimum PSF stars={}".format(min_psfstars))
    else:
        min_psfstars = args.min_psfstars

    for exposure in exposures:
        psfstars_count = len(exposure.get_catalog("psfstars.list").df)

        if psfstars_count < min_psfstars:
            flagged.append(exposure.name)

    lightcurve.add_noprocess(flagged)
    logger.info("{} exposures flagged has having PSF stars count < {}".format(len(flagged), args.min_psfstars))

    return True


def filter_astro_chi2(lightcurve, logger, args):
    import pandas as pd
    import numpy as np

    chi2_df = pd.read_csv(lightcurve.astrometry_path.joinpath("ref2px_chi2_exposures.csv"), index_col=0)

    to_filter = (np.any([chi2_df['chi2'] >= args.astro_max_chi2, chi2_df['chi2'].isna()], axis=0))

    logger.info("{} quadrants flagged as having astrometry chi2 >= {} or NaN values.".format(sum(to_filter), args.astro_max_chi2))
    logger.info("List of filtered quadrants:")
    logger.info(chi2_df.loc[to_filter].index)
    lightcurve.add_noprocess(chi2_df.loc[to_filter].index)

    return True


def filter_seeing(lightcurve, logger, args):
    # quadrant_paths = quadrants_from_band_path(band_path, logger)
    exposures = lightcurve.get_exposures()
    flagged = []

    for exposure in exposures:
        try:
            seeing = float(exposure.exposure_header['SEEING'])
        except KeyError as e:
            logger.error("{} - {}".format(exposure.name, e))
            continue

        if seeing > args.max_seeing:
            flagged.append(exposure.name)

    lightcurve.add_noprocess(flagged)
    logger.info("{} quadrants flagged as having seeing > {}".format(len(flagged), args.max_seeing))

    return True


def discard_calibrated(exposure, logger, args):
    return True


discard_calibrated_rm = ["calibrated.fits", "weight.fz"]


def catalogs_to_ds9regions(exposure, logger, args):
    import numpy as np

    def _write_ellipses(catalog_df, region_path, color='green'):
        with open(region_path, 'w') as f:
            f.write("global color={} dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n".format(color))
            f.write("image\n")

            for idx, x in catalog_df.iterrows():
                if 'num' in catalog_df.columns:
                    f.write("text {} {} {{{}}}\n".format(x['x']+1, x['y']+15, "{} {:.2f}-{:.2f}".format(int(x['num']), np.sqrt(x['gmxx']), np.sqrt(x['gmyy']))))
                f.write("ellipse {} {} {} {} {} # width=2\n".format(x['x']+1, x['y']+1, x['a'], x['b'], x['angle']))
                f.write("circle {} {} {}\n".format(x['x']+1, x['y']+1, 5.))

    def _write_circles(catalog_df, region_path, radius=10., color='green'):
        with open(region_path, 'w') as f:
            f.write("global color={} dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n".format(color))
            f.write("image\n")

            for idx, x in catalog_df.iterrows():
                f.write("circle {} {} {}\n".format(x['x']+1, x['y']+1, radius))

    #catalog_names = ["aperse", "standalone_stars", "se", "psfstars"]
    catalog_names = {"aperse": 'ellipse', "standalone_stars": 'ellipse', "psfstars": 'circle', "se": 'circle'}

    for catalog_name in catalog_names.keys():
        try:
            catalog_df = exposure.get_catalog(catalog_name + ".list").df
        except FileNotFoundError as e:
            logger.error("Could not found catalog {}!".format(catalog_name))
        else:
            region_path = exposure.path.joinpath("{}.reg".format(catalog_name))
            if catalog_names[catalog_name] == 'circle':
                _write_circles(catalog_df, region_path)
            else:
                _write_ellipses(catalog_df, region_path)
            # cat_to_ds9regions(catalog_df, exposure.path.joinpath("{}.reg".format(catalog_name))) #

    return True


def dummy(lightcurve, logger, args):
    return True


def concat_catalogs(lightcurve, logger, args):
    catalog = lightcurve.extract_star_catalog(['aperstars'])[['exposure', 'x', 'y', 'gmxx', 'gmyy', 'gmxy', 'apfl5', 'rad5']].rename(columns={'apfl5': 'aperflux', 'rad5': 'aperrad'})
    catalog['ccdid'] = catalog.apply(lambda x: int(x['exposure'][30:32]), axis=1)
    catalog['qid'] = catalog.apply(lambda x: int(x['exposure'][36]), axis=1)
    catalog.to_parquet("bigcat.parquet")
    return True
