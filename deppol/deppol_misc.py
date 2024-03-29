#!/usr/bin/env python3


def psf_study(quadrant_path, ztfname, filtercode, logger, args):
    from utils import ListTable, match_pixel_space, quadrant_size_px
    import numpy as np
    from saunerie.plottools import binplot
    from scipy.stats import norm
    import matplotlib
    import matplotlib.pyplot as plt
    import pickle
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    matplotlib.use('Agg')

    if not quadrant_path.joinpath("psfstars.list").exists():
        return True

    psf_resid = ListTable.from_filename(quadrant_path.joinpath("psf_resid_tuple.dat"))
    psf_tuple = ListTable.from_filename(quadrant_path.joinpath("psftuple.list"))
    psf_stars = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))
    stand = ListTable.from_filename(quadrant_path.joinpath("standalone_stars.list"))

    stand.df['mag'] = -2.5*np.log10(stand.df['flux'])
    psf_stars.df['mag'] = -2.5*np.log10(psf_stars.df['flux'])
    psf_residuals_px = (psf_resid.df['fimg'] - psf_resid.df['fpsf']).to_numpy()
    mag = np.linspace(np.min(stand.df['mag']), np.max(stand.df['mag']), 100)

    subx = 4
    suby = 1

    # Fit order 2 polynomials on skewness/mag relation
    def _fit_skewness_subquadrants(axis, sub):
        ranges = np.linspace(0., quadrant_size_px[axis], sub+1)
        skew_polynomials = []
        for i in range(sub):
            range_mask = (stand.df['y'] >= ranges[i]) & (stand.df['y'] < ranges[i+1])
            skew_polynomials.append(np.polynomial.Polynomial.fit(stand.df.loc[range_mask]['mag'], stand.df.loc[range_mask]['gm{}3'.format(axis)], 1))

        return skew_polynomials

    skew_polynomial_x = np.polynomial.Polynomial.fit(stand.df['mag'], stand.df['gmx3'], 1)
    skew_polynomial_y = np.polynomial.Polynomial.fit(stand.df['mag'], stand.df['gmy3'], 1)

    skew_polynomial_subx = _fit_skewness_subquadrants('x', subx)
    skew_polynomial_suby = _fit_skewness_subquadrants('y', suby)

    with open(quadrant_path.joinpath("stamp_skewness.pickle"), 'wb') as f:
        pickle.dump({'poly_x': skew_polynomial_x, 'poly_y': skew_polynomial_y,
                     'poly_x_sub': skew_polynomial_subx, 'poly_y_sub': skew_polynomial_suby}, f)

    plt.figure(figsize=(7., 7.))
    ax = plt.axes()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.suptitle("Skewness plane for standalone stars stamps in \n {}".format(quadrant_path.name))
    plt.scatter(stand.df['gmx3'], stand.df['gmy3'], s=2., color='black')
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
    plt.scatter(stand.df['mag'], stand.df['gmx3'], s=2.)
    plt.plot(mag, skew_polynomial_x(mag), color='black')
    plt.axhline(0.)
    plt.ylim(-0.5, 0.3)
    plt.xlim(-15., -7)
    plt.ylabel("$M^g_{xxx}$")

    ax = plt.subplot(2, 1, 2)
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(linestyle='--', color='xkcd:sky blue')
    plt.scatter(stand.df['mag'], stand.df['gmy3'], s=2.)
    plt.plot(mag, skew_polynomial_y(mag), color='black')
    plt.axhline(0.)
    plt.xlabel("$m$")
    plt.ylim(-0.3, 0.3)
    plt.xlim(-15., -6)
    plt.ylabel("$M^g_{yyy}$")

    plt.savefig(quadrant_path.joinpath("{}_skewness_magnitude.png".format(quadrant_path.name)), dpi=300.)
    plt.close()
    return

    star_indices = match_pixel_space(psf_stars.df[['x', 'y']].to_records(), psf_resid.df[['xc', 'yc']].rename(mapper={'xc': 'x', 'yc': 'y'}, axis='columns').to_records())
    psf_size = int(np.sqrt(sum(star_indices==0)))
    psf_residuals = np.zeros([len(psf_stars.df), psf_size*psf_size])
    for star_index in range(len(np.bincount(star_indices))):
        star_mask = (star_indices==star_index)
        ij = (psf_resid.df.iloc[star_mask]['j']*psf_size + psf_resid.df.iloc[star_mask]['i'] + int(np.floor(psf_size**2/2))).to_numpy()
        np.put_along_axis(psf_residuals[star_index], ij, psf_residuals_px[star_mask], 0)

    psf_residuals = psf_residuals.reshape(len(psf_stars.df), psf_size, psf_size)

    bins_count = 5
    mag_range = (psf_stars.df['mag'].min(), psf_stars.df['mag'].max())
    bins = np.linspace(*mag_range, bins_count+1)
    psf_residual_means = []
    psf_residual_count = []
    for i in range(bins_count):
        lower_bound = bins[i]
        upper_bound = bins[i+1]

        binned_stars_mask = np.all([psf_stars.df['mag'] < upper_bound, psf_stars.df['mag'] >= lower_bound], axis=0)
        psf_residual_means.append(np.mean(psf_residuals[binned_stars_mask], axis=0))
        psf_residual_count.append(sum(binned_stars_mask))

    plt.subplots(ncols=bins_count, nrows=1, figsize=(5.*bins_count, 5.))
    plt.suptitle("Binned PSF residuals for {}".format(quadrant_path.name))
    for i, psf_residual_mean in enumerate(psf_residual_means):
        plt.subplot(1, bins_count,  1+i)
        plt.imshow(psf_residual_mean)
        plt.axis('off')
        plt.title("${0:.2f} \leq m < {1:.2f}$\n$N={2}$".format(bins[i], bins[i+1], psf_residual_count[i]))

    plt.savefig(quadrant_path.joinpath("{}_psf_residuals.png".format(quadrant_path.name)))
    plt.close()
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


def seeing_study(band_path, ztfname, filtercode, logger, args):
    from utils import get_header_from_quadrant_path, plot_ztf_focal_plan_values, ListTable
    from deppol_utils import quadrants_from_band_path
    import pickle
    import numpy as np
    import itertools
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    quadrant_paths = quadrants_from_band_path(band_path, logger, check_files="psfstars.list", ignore_noprocess=False)

    keys = {'seeing': 'hdr',
            'seseeing': 'hdr',
            'gfseeing': 'gfseeing',
            'momentx1': 'psfstars',
            'momenty1': 'psfstars',
            'momentx2': 'psfstars',
            'momenty2': 'psfstars',
            'momentxy': 'psfstars'}

    if not band_path.joinpath("seeings.pickle").exists() or args.from_scratch:
        vals = {}

        for key in keys:
            vals[key] = {}
            for i in range(1, 17):
                vals[key][i] = {}
                for j in range(4):
                    vals[key][i][j] = []

        for quadrant_path in quadrant_paths:
            hdr = get_header_from_quadrant_path(quadrant_path)
            psfstars_list = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))

            for key in keys.keys():
                if keys[key] == 'psfstars':
                    vals[key][hdr['ccdid']][hdr['qid']-1].append(psfstars_list.header[key])
                elif key == 'seseeing':
                    vals[key][hdr['ccdid']][hdr['qid']-1].append(2.355*hdr[key])
                elif keys[key] == 'hdr':
                    vals[key][hdr['ccdid']][hdr['qid']-1].append(hdr[key])
                elif key == 'gfseeing':
                    with open(quadrant_path.joinpath("psf.dat"), 'r') as f:
                        lines = f.readlines()
                        gfseeing = float(lines[3].split(" ")[0].strip())
                    vals[key][hdr['ccdid']][hdr['qid']-1].append(2.355*gfseeing)

        with open(band_path.joinpath("seeings.pickle"), 'wb') as f:
            pickle.dump(vals, f)
    else:
        with open(band_path.joinpath("seeings.pickle"), 'rb') as f:
            vals = pickle.load(f)

    for key in keys:
        for key_i in vals[key].keys():
            for key_j in vals[key][key_i].keys():
                vals[key][key_i][key_j] = np.median(list(map(lambda x: float(x), vals[key][key_i][key_j])))

    cmap = 'coolwarm'
    def _plot_focal_plane_seeing(key):
        val_list = [[vals[key][key_i][key_j] for key_j in vals[key][key_i].keys()] for key_i in vals[key].keys()]
        val_list = list(itertools.chain(*val_list))
        vmin = min(val_list)
        vmax = max(val_list)

        cm = ScalarMappable(cmap=cmap)
        cm.set_clim(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(5., 6.), constrained_layout=False)
        #f1, f2, f3 = fig.subfigures(ncols=1, nrows=3, height_ratios=[12., 1., 1.])
        f1, f2 = fig.subfigures(ncols=1, nrows=2, height_ratios=[12., 0.3], wspace=-1., hspace=-1.)
        f1.suptitle("\n{}-{} Focal plane {}".format(ztfname, filtercode, key), fontsize='large')
        plot_ztf_focal_plan_values(f1, vals[key], vmin=vmin, vmax=vmax, cmap=cmap, scalar=True)
        ax2 = f2.add_subplot(clip_on=False)
        f2.subplots_adjust(bottom=-5, top=5.)
        ax2.set_position((0.2, 0, 0.6, 1))
        # ax2 = f2.add_subplot()
        # ax3 = f3.add_subplot(axis='none')
        # Colorbar(ax, cm, orientation='horizontal')
        cb = f1.colorbar(cm, cax=ax2, orientation='horizontal')
        # ax2.autoscale(tight=True)
        ax2.set_clip_on(False)
        ax2.minorticks_on()
        ax2.tick_params(direction='inout')
        plt.savefig(band_path.joinpath("{}-{}_focal_plane_{}.png".format(ztfname, filtercode, key)), dpi=300., bbox_inches='tight', pad_inches=0.1)
        plt.close()

    for key in keys:
        # print("Plotting {}".format(key))
        _plot_focal_plane_seeing(key)


def match_gaia(quadrant_path, ztfname, filtercode, logger, args):
    from itertools import chain
    from utils import read_list, get_wcs_from_quadrant, get_mjd_from_quadrant_path, contained_in_exposure, match_pixel_space, gaiarefmjd, quadrant_width_px, quadrant_height_px
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not quadrant_path.joinpath("psfstars.list").exists():
        return False

    _, stars_df = read_list(quadrant_path.joinpath("psfstars.list"))
    _, aper_stars_df = read_list(quadrant_path.joinpath("standalone_stars.list"))
    wcs = get_wcs_from_quadrant(quadrant_path)
    obsmjd = get_mjd_from_quadrant_path(quadrant_path)
    center_radec = wcs.pixel_to_world_values(np.array([[quadrant_width_px/2., quadrant_height_px/2.]]))[0]
    center_skycoord = SkyCoord(ra=center_radec[0], dec=center_radec[1], unit='deg')

    # Match aperture photometry and psf photometry catalogs
    i = match_pixel_space(stars_df[['x', 'y']], aper_stars_df[['x', 'y']], radius=1.)
    stars_df = stars_df.iloc[i[i>=0]].reset_index(drop=True)
    aper_stars_df = aper_stars_df.iloc[i>=0].reset_index(drop=True)

    # Add aperture photometry
    aper_fields = [['apfl'+str(i), 'eapfl'+str(i), 'rad'+str(i)] for i in range(10)]
    aper_fields = list(chain(*aper_fields))
    stars_df[aper_fields] = aper_stars_df[aper_fields]

    gaia_stars_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['gaiaid'] = gaia_stars_df.index

    # First order proper motion correction for indentification
    # We do it in astrometry code
    # gaia_stars_df['ra'] = gaia_stars_df['ra']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmra']/np.cos(gaia_stars_df['dec']/180.*np.pi)/1000./3600./365.25
    # gaia_stars_df['dec'] = gaia_stars_df['dec']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmde']/1000./3600./365.25
    # mas/year -> deg/day
    gaia_stars_df['pmra'] = gaia_stars_df['pmra']/np.cos(np.deg2rad(gaia_stars_df['dec']))/1000./3600./365.25
    gaia_stars_df['pmdec'] = gaia_stars_df['pmdec']/1000./3600./365.25
    gaia_stars_df.dropna(inplace=True)

    ra_corrected = gaia_stars_df['ra']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmra']
    dec_corrected = gaia_stars_df['dec']+(obsmjd-gaiarefmjd)*gaia_stars_df['pmdec']

    # gaia_stars_radec = SkyCoord(gaia_stars_df['ra'], gaia_stars_df['dec'], unit='deg')
    gaia_stars_skycoords = SkyCoord(ra_corrected, dec_corrected, unit='deg')

    sep = center_skycoord.separation(gaia_stars_skycoords)
    idxc = (sep < 0.5*u.deg)
    gaia_stars_skycoords = gaia_stars_skycoords[idxc]
    gaia_stars_df = gaia_stars_df[idxc]

    #gaia_mask = contained_in_exposure(gaia_stars_radec, wcs, return_mask=True)
    #gaia_stars_df = gaia_stars_df.iloc[gaia_mask]
    #x, y = gaia_stars_radec[gaia_mask].to_pixel(wcs)
    x, y = gaia_stars_skycoords.to_pixel(wcs)
    gaia_stars_df['x'] = x
    gaia_stars_df['y'] = y
    gaia_stars_df = gaia_stars_df.dropna()

    try:
        i = match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), stars_df[['x', 'y']].to_records(), radius=2.)
    except Exception as e:
        logger.error("Could not match any Gaia stars!")
        raise ValueError("Could not match any Gaia stars!")

    matched_gaia_stars_df = gaia_stars_df.iloc[i[i>=0]].reset_index(drop=True)
    matched_stars_df = stars_df.iloc[i>=0].reset_index(drop=True)
    logger.info("Matched {} GAIA stars".format(len(matched_gaia_stars_df)))

    with pd.HDFStore(quadrant_path.joinpath("matched_stars.hd5"), 'w') as hdfstore:
        hdfstore.put('matched_gaia_stars', matched_gaia_stars_df)
        hdfstore.put('matched_stars', matched_stars_df)

    return True


def match_gaia_reduce(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    import numpy as np
    from utils import get_header_from_quadrant_path
    from deppol_utils import quadrants_from_band_path

    quadrant_paths = quadrants_from_band_path(band_path, logger, check_files="match_gaia.success")

    logger.info("Concatenating matched stars measurements from {} quadrants.".format(len(quadrant_paths)))

    matched_stars_list = []
    quadrants_dict = {}

    for quadrant_path in quadrant_paths:
        logger.info(quadrant_path.name)
        matched_gaia_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_gaia_stars')
        matched_stars_df = pd.read_hdf(quadrant_path.joinpath("matched_stars.hd5"), key='matched_stars')

        matched_gaia_stars_df.rename(columns={'x': 'gaia_x', 'y': 'gaia_y'}, inplace=True)
        matched_stars_df['mag'] = -2.5*np.log10(matched_stars_df['flux'])
        matched_stars_df['emag'] = matched_stars_df['eflux']/matched_stars_df['flux']

        matched_stars_df = pd.concat([matched_stars_df, matched_gaia_stars_df], axis=1)

        quadrant_dict = {}
        header = get_header_from_quadrant_path(quadrant_path)
        quadrant_dict['quadrant'] = quadrant_path.name
        quadrant_dict['airmass'] = header['airmass']
        quadrant_dict['mjd'] = header['obsmjd']
        quadrant_dict['seeing'] = header['seeing']
        quadrant_dict['ha'] = header['hourangd'] #*15
        quadrant_dict['ha_15'] = 15.*header['hourangd']
        quadrant_dict['lst'] = header['oblst']
        quadrant_dict['azimuth'] = header['azimuth']
        quadrant_dict['dome_azimuth'] = header['dome_az']
        quadrant_dict['elevation'] = header['elvation']
        quadrant_dict['z'] = 90. - header['elvation']
        quadrant_dict['telra'] = header['telrad']
        quadrant_dict['teldec'] = header['teldecd']

        quadrant_dict['field'] = header['dbfield']
        quadrant_dict['ccdid'] = header['ccd_id']
        quadrant_dict['qid'] = header['amp_id']
        quadrant_dict['rcid'] = header['dbrcid']

        quadrant_dict['fid'] = header['dbfid']

        quadrant_dict['temperature'] = header['tempture']
        quadrant_dict['head_temperature'] = header['headtemp']
        quadrant_dict['ccdtemp'] = float(header['ccdtmp{}'.format(str(header['ccd_id']).zfill(2))])

        quadrant_dict['wind_speed'] = header['windspd']
        quadrant_dict['wind_dir'] = header['winddir']
        quadrant_dict['dewpoint'] = header['dewpoint']
        quadrant_dict['humidity'] = header['humidity']
        quadrant_dict['wetness'] = header['wetness']
        quadrant_dict['pressure'] = header['pressure']

        quadrant_dict['crpix1'] = header['crpix1']
        quadrant_dict['crpix2'] = header['crpix2']
        quadrant_dict['crval1'] = header['crval1']
        quadrant_dict['crval2'] = header['crval2']
        quadrant_dict['cd_11'] = header['cd1_1']
        quadrant_dict['cd_12'] = header['cd1_2']
        quadrant_dict['cd_21'] = header['cd2_1']
        quadrant_dict['cd_22'] = header['cd2_2']

        quadrant_dict['skylev'] = header['sexsky']
        quadrant_dict['sigma_skylev'] = header['sexsigma']

        for key in quadrant_dict.keys():
            matched_stars_df[key] = quadrant_dict[key]

        matched_stars_list.append(matched_stars_df)
        quadrants_dict[quadrant_path.name] = quadrant_dict

    matched_stars_df = pd.concat(matched_stars_list, axis=0, ignore_index=True)

    # Build gaia star catalog
    gaia_stars_df = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(ztfname)), key='gaia_cal')
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['gaiaid'] = gaia_stars_df.index
    gaia_stars_df = gaia_stars_df[~gaia_stars_df.index.duplicated(keep='first')]

    gaia_stars = []
    for gaiaid in set(matched_stars_df['gaiaid']):
        #gaia_stars.append(pd.Series(gaia_stars_df.drop(['rcid', 'field'], axis='columns').loc[gaiaid], name=gaiaid))
        gaia_stars.append(pd.Series(gaia_stars_df.loc[gaiaid], name=gaiaid))

    gaia_stars_df = pd.DataFrame(data=gaia_stars)
    gaia_stars_df.rename({'pmde': 'pmdec'}, axis='columns', inplace=True)
    gaia_stars_df['pmra'] = gaia_stars_df['pmra']/np.cos(np.deg2rad(gaia_stars_df['dec']))/1000./3600./365.25
    gaia_stars_df['pmdec'] = gaia_stars_df['pmdec']/1000./3600./365.25

    # Remove measures with Nan's
    nan_mask = matched_stars_df.isna().any(axis=1)
    matched_stars_df = matched_stars_df[~nan_mask]
    logger.info("Removed {} measurements with NaN's".format(sum(nan_mask)))

    nan_mask = gaia_stars_df.isna().any(axis=1)
    gaia_stars_df = gaia_stars_df[~nan_mask]
    logger.info("Removed {} Gaia stars with NaN's".format(sum(nan_mask)))

    # Compute color
    matched_stars_df['colormag'] = matched_stars_df['bpmag'] - matched_stars_df['rpmag']
    gaia_stars_df['colormag'] = gaia_stars_df['bpmag'] - gaia_stars_df['rpmag']

    # Save to disk
    matched_stars_df.to_parquet(band_path.joinpath("matched_stars.parquet"))
    logger.info("Total matched Gaia stars: {}".format(len(matched_stars_df)))

    quadrants_df = pd.DataFrame.from_dict(quadrants_dict, orient='index')
    quadrants_df.to_parquet(band_path.joinpath("quadrants.parquet"))

    gaia_stars_df.to_parquet(band_path.joinpath("gaia_stars.parquet"))

    return True


# Extract data from standalone stars and plot several distributions
def stats(quadrant_path, ztfname, filtercode, logger, args):
    import warnings
    import pandas as pd
    from utils import read_list
    from astropy.io import fits

    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

    import pandas as pd

    def _extract_from_list(list_filename, hdfstore):
        list_path = quadrant_path.joinpath(list_filename).with_suffix(".list")

        if not list_path.exists():
            return False

        with open(list_path, mode='r') as f:
            global_params, df = read_list(f)

        hdfstore.put(list_path.stem, df)
        hdfstore.put("{}_globals".format(list_path.stem), pd.DataFrame([global_params]))

        return True

    with pd.HDFStore(quadrant_path.joinpath("lists.hdf5"), mode='w') as hdfstore:
        # From make_catalog
        _extract_from_list("se", hdfstore)

        # From mkcat2
        cont = _extract_from_list("standalone_stars", hdfstore)

        if not cont:
            return True

        _extract_from_list("aperse", hdfstore)

        # From calibrated.fits
        keywords = ['sexsky', 'sexsigma', 'bscale', 'bzero', 'origsatu', 'saturlev', 'backlev', 'back_sub', 'seseeing', 'gfseeing']

        calibrated = {}
        with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
            for keyword in keywords:
                calibrated[keyword] = hdul[0].header[keyword]

            hdfstore.put('calibrated', pd.DataFrame([calibrated]))

        # From makepsf
        cont = _extract_from_list("psfstars", hdfstore)

        if not cont:
            return True

        _extract_from_list("psftuples", hdfstore)

    return True


def stats_reduce(band_path, ztfname, filtercode, logger, args):
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from deppol_utils import quadrants_from_band_path
    from utils import get_header_from_quadrant_path, ListTable
    matplotlib.use('Agg')

    # Seeing histogram
    folders = [folder for folder in band_path.glob("*") if folder.is_dir()]
    quadrant_paths = quadrants_from_band_path(band_path, logger, ignore_noprocess=True)

    logger.info("Plotting fitted seeing histogram")
    seeings = []
    for quadrant_path in quadrant_paths:
        if quadrant_path.joinpath("calibrated.fits").exists():
           hdr = get_header_from_quadrant_path(quadrant_path)
           seeings.append(hdr['seeing'])

    ax = plt.axes()
    plt.suptitle("Seeing distribution (computed by the ZTF pipeline)")
    #plt.hist(seeings, bins=int(len(seeings)/4), color='xkcd:dark grey', histtype='step')
    plt.hist(seeings, bins='auto', color='xkcd:dark grey', histtype='step')
    if args.max_seeing:
        removed_quadrants = len(list(filter(lambda x: x >= args.max_seeing, seeings)))
        plt.axvline(args.max_seeing)
        plt.text(0.2, 0.9, "{:.02f}% quadrants with seeing > {} ({} quadrants)".format(removed_quadrants/len(seeings)*100, args.max_seeing, len(seeings)), transform=ax.transAxes)
    plt.grid()
    plt.xlabel("Seeing FWHM [pixel]")
    plt.ylabel("Count")
    plt.savefig(band_path.joinpath("{}-{}_seeing_dist.png".format(ztfname, filtercode)), dpi=300)
    plt.close()

    quadrant_paths = quadrants_from_band_path(band_path, logger, check_files=['psfstars.list', 'calibrated.fits'], ignore_noprocess=True)
    logger.info("Plotting psfstars count vs seeing")

    seeings = []
    psfstars_counts = []
    for quadrant_path in quadrant_paths:
        hdr = get_header_from_quadrant_path(quadrant_path)
        seeings.append(hdr['seeing'])

        psfstars_list = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))
        psfstars_counts.append(len(psfstars_list.df))

    ax = plt.axes()
    plt.plot(seeings, psfstars_counts, '.')

    if args.min_psfstars:
        filtered_psfstars_count = len(list(filter(lambda x: x < args.min_psfstars, psfstars_counts)))
        plt.axhline(args.min_psfstars)
        plt.text(0.2, 0.1, "{:.02f}% quadrants ({}) with PSF stars count < {}".format(filtered_psfstars_count/len(quadrant_paths)*100., filtered_psfstars_count, args.min_psfstars), transform=ax.transAxes)

    if args.max_seeing:
        plt.axvline(args.max_seeing)

    plt.grid()
    plt.xlabel("Seeing FWHM [pixel]")
    plt.ylabel("PSF stars count")
    plt.savefig(band_path.joinpath("{}-{}_psfstars_seeing.png".format(ztfname, filtercode)), dpi=300.)
    plt.close()

    with open(band_path.joinpath("{}-{}_failures.txt".format(ztfname, filtercode)), 'w') as f:
        # Failure rates
        def _failure_rate(listname, func):
            success_count = 0
            for quadrant_path in quadrant_paths:
                if quadrant_path.joinpath("{}.list".format(listname)).exists():
                    success_count += 1

            f.writelines(["For {}:\n".format(func),
                          " Success={}/{}\n".format(success_count, len(folders)),
                          " Rate={}\n\n".format(float(success_count)/len(folders))])

        _failure_rate("se", 'make_catalog')
        _failure_rate("standalone_stars", 'mkcat2')
        _failure_rate("psfstars", 'makepsf')

    logger.info("Plotting computing time histograms")
    # Plot results_*.csv histogram
    result_paths = list(band_path.glob("results_*.csv"))
    for result_path in result_paths:
        func = "_".join(str(result_path.stem).split("_")[1:])
        result_df = pd.read_csv(result_path)
        computation_times = (result_df['time_end'] - result_df['time_start']).to_numpy()
        plt.hist(computation_times, bins=int(len(result_df)/4), histtype='step')
        plt.xlabel("Computation time (s)")
        plt.ylabel("Count")
        plt.title("Computation time for {}".format(func))
        plt.grid()
        plt.savefig(band_path.joinpath("{}-{}_{}_compute_time_dist.png".format(ztfname, filtercode, func)), dpi=300)
        plt.close()


def clean(quadrant_path, ztfname, filtercode, logger, args):
    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["elixir.fits", "dead.fits.gz", ".dbstuff"]
    files_to_delete = list(filter(lambda f: f.name not in files_to_keep, list(quadrant_path.glob("*"))))

    for file_to_delete in files_to_delete:
            file_to_delete.unlink()

    return True


def clean_reduce(band_path, ztfname, filtercode, logger, args):
    from shutil import rmtree

    # We want to delete all files in order to get back to the prepare_deppol stage
    files_to_keep = ["prepare.log"]

    # Delete all files
    files_to_delete = list(filter(lambda f: f.is_file() and (f.name not in files_to_keep), list(band_path.glob("*"))))

    [f.unlink() for f in files_to_delete]

    # Delete output folders
    # We delete everything but the quadrant folders
    folders_to_keep = list(band_path.glob("ztf_*"))
    folders_to_delete = [folder for folder in list(band_path.glob("*")) if (folder not in folders_to_keep) or folder.is_file()]

    [rmtree(band_path.joinpath(folder), ignore_errors=True) for folder in folders_to_delete]


def filter_psfstars_count(band_path, ztfname, filtercode, logger, args):
    from deppol_utils import quadrants_from_band_path, noprocess_quadrants
    from utils import ListTable

    quadrant_paths = quadrants_from_band_path(band_path, logger, ignore_noprocess=True)
    noprocess = noprocess_quadrants(band_path)
    flagged_count = 0
    quadrants_to_flag = []

    for quadrant_path in quadrant_paths:
        if quadrant_path.joinpath("psfstars.list").exists():
            quadrant = quadrant_path.name
            psfstars_list = ListTable.from_filename(quadrant_path.joinpath("psfstars.list"))
            psfstars_count = len(psfstars_list.df)

            if psfstars_count <= args.min_psfstars:
                flagged_count += 1

                if quadrant not in noprocess:
                    quadrants_to_flag.append(quadrant)

    if quadrants_to_flag:
        with open(band_path.joinpath("noprocess"), 'a') as f:
            for quadrant_to_flag in quadrants_to_flag:
                f.write("{}\n".format(quadrant_to_flag))

    flagged_count = len(quadrants_to_flag)
    logger.info("{} quadrants flagged as having PSF stars count <= {}.".format(flagged_count, args.min_psfstars))
    logger.info("{} quadrants added to the noprocess list.".format(len(quadrants_to_flag)))

    return True


def filter_astro_chi2(band_path, ztfname, filtercode, logger, args):
    from deppol_utils import noprocess_quadrants
    import pandas as pd
    import numpy as np

    noprocess = noprocess_quadrants(band_path)

    chi2_path = band_path.joinpath("astrometry/ref2px_chi2_quadrants.csv")
    if not chi2_path.exists():
        logger.info("Could not find file {}!".format(chi2_path))
        return False

    chi2_df = pd.read_csv(chi2_path, index_col=0)

    to_filter = (np.any([chi2_df['chi2'] >= args.astro_max_chi2, chi2_df['chi2'].isna()], axis=0))
    # to_filter = (np.any([chi2_df['chi2'].isna()], axis=0))
    logger.info("{} quadrants flagged as having astrometry chi2 >= {} or NaN values.".format(sum(to_filter), args.astro_max_chi2))
    logger.info("List of filtered quadrants:")
    logger.info(chi2_df.loc[to_filter])

    quadrants_to_flag = 0
    with open(band_path.joinpath("noprocess"), 'a') as f:
        for quadrant in chi2_df.loc[to_filter].index.tolist():
            if quadrant not in noprocess:
                f.write("{}\n".format(quadrant))
                quadrants_to_flag += 1

    logger.info("{} quadrants added to the noprocess list.".format(quadrants_to_flag))

    return True


def filter_seeing(band_path, ztfname, filtercode, logger, args):
    from utils import get_header_from_quadrant_path
    from deppol_utils import quadrants_from_band_path, noprocess_quadrants, run_and_log

    quadrant_paths = quadrants_from_band_path(band_path, logger)
    flagged_count = 0
    quadrants_to_flag = []
    noprocess = noprocess_quadrants(band_path)

    for quadrant_path in quadrant_paths:
        quadrant = quadrant_path.name

        try:
            seeing = get_header_from_quadrant_path(quadrant_path)['SEEING']
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Folder content:")
            run_and_log(["ls", "-lah", quadrant_path], logger)
            continue
        except KeyError as e:
            logger.error(quadrant)
            logger.error(e)
            quadrants_to_flag.append(quadrant)
            continue

        if seeing > args.max_seeing:
            flagged_count += 1

            if quadrant not in noprocess:
                quadrants_to_flag.append(quadrant)

    if quadrants_to_flag:
        with open(band_path.joinpath("noprocess"), 'a') as f:
            for quadrant_to_flag in quadrants_to_flag:
                f.write("{}\n".format(quadrant_to_flag))

    logger.info("{} quadrants flagged as having seeing > {}.".format(flagged_count, args.max_seeing))
    logger.info("{} quadrants added to the noprocess list.".format(len(quadrants_to_flag)))

    return True


def discard_calibrated(quadrant_path, ztfname, filtercode, logger, args):
    pass


discard_calibrated_rm = ["calibrated.fits", "weight.fz"]
