#!/usr/bin/env python3

def psfstudy(band_path, ztfname, filtercode, logger, args):
    import pathlib
    import gc
    import traceback

    import numpy as np
    from numpy.polynomial.polynomial import Polynomial
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from saunerie.plottools import binplot
    import pandas as pd
    from dask import delayed, compute

    from deppol_utils import quadrants_from_band_path
    from utils import get_header_from_quadrant_path, plot_ztf_focal_plan_values, ListTable, match_pixel_space, quadrant_name_explode

    matplotlib.use('Agg')

    output_path = band_path.joinpath("psfstudy")
    output_path.mkdir(exist_ok=True)

    quadrant_paths = quadrants_from_band_path(band_path, logger)

    # Remove bad QC quadrants
    def _quadrant_good(quadrant_path):
        if not quadrant_path.joinpath("match_gaia.success").exists():
            return False

        if len(ListTable.from_filename(quadrant_path.joinpath("psfstars.list")).df) < 150:
            return False

        return True

    print("Found {} quadrants".format(len(quadrant_paths)))
    # quadrant_paths = list(filter(_quadrant_good, quadrant_paths))
    # print("After QC: {}".format(len(quadrant_paths)))

    min_mag, max_mag = 13., 19.
    low_bin, high_bin = 14, 17

    def _process_quadrant(quadrant_path):
        if not _quadrant_good(quadrant_path):
            return

        try:
            quadrant_header = get_header_from_quadrant_path(quadrant_path)
            expid = int(quadrant_header['expid'])
            mjd = float(quadrant_header['obsmjd'])
            ccdid = int(quadrant_header['ccdid'])
            qid = int(quadrant_header['qid'])
            skylev = float(quadrant_header['sexsky'])
            moonillf = float(quadrant_header['moonillf'])

            year, month, day, _, _, ccdid, qid = quadrant_name_explode(quadrant_path.name)

            exposure_output_path = output_path.joinpath("{}-{}-{}-{}".format(year, month, day, expid))
            exposure_output_path.mkdir(exist_ok=True)

            # First load PSF and Gaia catalog
            with pd.HDFStore(quadrant_path.joinpath("matched_stars.hd5")) as hdfstore:
                cat_gaia = hdfstore.get('/matched_gaia_stars')
                cat_psfstars = hdfstore.get('/matched_stars')

            # Match aperture catalog to Gaia
            cat_aperstars = ListTable.from_filename(quadrant_path.joinpath("standalone_stars.list")).df

            i = match_pixel_space(cat_gaia[['x', 'y']].to_records(), cat_aperstars[['x', 'y']].to_records(), radius=1.)

            cat_psfstars = cat_psfstars.iloc[i[i>=0]].reset_index(drop=True)
            cat_gaia = cat_gaia.iloc[i[i>=0]].reset_index(drop=True)
            cat_aperstars = cat_aperstars.iloc[i>=0].reset_index(drop=True)

            # Convert ADU flux into mag
            cat_psfstars['mag'] = -2.5*np.log10(cat_psfstars['flux'])
            cat_aperstars['mag'] = -2.5*np.log10(cat_aperstars['apfl5'])
            cat_psfstars['emag'] = -1.08*cat_psfstars['eflux']/cat_psfstars['flux']
            cat_aperstars['emag'] = -1.08*cat_aperstars['eapfl5']/cat_aperstars['apfl5']
            aper_size = list(set(cat_aperstars['rad5']))[0]

            # Compute delta flux between PSF and aperture photometry
            deltamag = (cat_psfstars['mag'] - cat_aperstars['mag']).to_numpy()
            deltamag = deltamag - np.mean(deltamag)
            edeltamag = np.sqrt(cat_psfstars['emag']**2+cat_aperstars['emag']**2).to_numpy()

            # Bin delta flux
            plt.figure(figsize=(7., 4.))
            gmag_binned, deltamag_binned, edeltamag_binned = binplot(cat_gaia['gmag'].to_numpy(), deltamag, robust=True, data=False, scale=True, weights=1./edeltamag**2, bins=np.linspace(min_mag, max_mag, 15), color='black', zorder=15)

            bin_mask = (edeltamag_binned > 0.)
            gmag_binned = gmag_binned[bin_mask]
            deltamag_binned = np.array(deltamag_binned)[bin_mask]
            edeltamag_binned = edeltamag_binned[bin_mask]

            # Fit polynomials on delta mag bins
            poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gmag_binned, deltamag_binned, 0, w=1./edeltamag_binned, full=True)
            poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gmag_binned, deltamag_binned, 1, w=1./edeltamag_binned, full=True)
            poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gmag_binned, deltamag_binned, 2, w=1./edeltamag_binned, full=True)

            poly0_chi2 = poly0_chi2/(len(gmag_binned)-1)
            poly1_chi2 = poly1_chi2/(len(gmag_binned)-2)
            poly2_chi2 = poly2_chi2/(len(gmag_binned)-3)

            # Compute rough delta mag on delta mag bins
            low_bin_idx = np.argmin(np.abs(gmag_binned-low_bin))
            high_bin_idx = np.argmin(np.abs(gmag_binned-high_bin))
            low_bin_indices = list(set([max([0, low_bin_idx-1]), low_bin_idx, low_bin_idx+1]))
            high_bin_indices = list(set([high_bin_idx-1, high_bin_idx, min([high_bin_idx+1, len(gmag_binned)-1])]))
            bin_low = np.average(deltamag_binned[low_bin_indices], weights=1./edeltamag_binned[low_bin_indices])
            bin_high = np.average(deltamag_binned[high_bin_indices], weights=1./edeltamag_binned[high_bin_indices])
            deltamag_bin = bin_high-bin_low

            # Start plot
            plt.title("Aperture - PSF [mag]\n({}-{}-{}) CCD={} Q={}\nAperture size={:.2f}\nSky level={:.2f}".format(year, month, day, ccdid, qid, aper_size, skylev))
            plt.grid()
            gmag_space = np.linspace(min_mag, max_mag, 100)
            plt.plot(gmag_space, poly0(gmag_space), ls='-', lw=1.5, color='grey', label="Constant - $\chi^2_\\nu={:.3f}$".format(poly0_chi2), zorder=10)
            plt.plot(gmag_space, poly1(gmag_space), ls='dotted', lw=1.5, color='grey', label="Linear - $\chi^2_\\nu={:.3f}$".format(poly1_chi2), zorder=10)
            plt.plot(gmag_space, poly2(gmag_space), ls='--', lw=1.5, color='grey', label="Quadratic - $\chi^2_\\nu={:.3f}$".format(poly2_chi2), zorder=10)
            plt.errorbar(cat_gaia['gmag'].to_numpy(), deltamag, yerr=edeltamag, lw=0., marker='.', markersize=1.5, zorder=0, color='xkcd:light blue')
            plt.legend(fontsize='small', loc='upper left')
            plt.xlabel("$g_\mathrm{Gaia}$ [mag]")
            plt.ylabel("$\Delta_M$ [mag]")
            plt.text(14., -0.1, "$\Delta M_{{{}-{}}}={:.5f}$".format(low_bin, high_bin, deltamag_bin), bbox=dict(boxstyle="round", ec='grey', fc='xkcd:light blue'))
            plt.xlim(min_mag, max_mag)
            plt.ylim(-0.2, 0.2)

            plt.tight_layout()
            plt.savefig(exposure_output_path.joinpath(quadrant_path.name), dpi=200.)
            plt.close()

            print(".", end="", flush=True)

            return {'quadrant': quadrant_path.name,
                    'date': "{}-{}-{}".format(year, month, day),
                    'mjd': mjd,
                    'expid': expid,
                    'ccdid': ccdid,
                    'qid': qid,
                    'skylev': skylev,
                    'deltamag': deltamag_bin,
                    'poly0_chi2': poly0_chi2,
                    'poly1_chi2': poly1_chi2,
                    'poly2_chi2': poly2_chi2}

        except Exception as e:
            print(quadrant_path)
            print(e)
            print(traceback.format_exc())


    # d = [_process_quadrant(quadrant_path) for quadrant_path in quadrant_paths]

    tasks = [delayed(_process_quadrant)(quadrant_path) for quadrant_path in quadrant_paths]
    results = compute(tasks)

    print("")
    print("Done.")
    results = list(filter(lambda x: x is not None, results[0]))
    df = pd.DataFrame(results)
    print(df)
    df.to_parquet(band_path.joinpath("myquadrants.parquet"))

        # for ccdid in range(1, 17):
        #     delta_mags[exposure][ccdid] = {}
        #     for qid in range(1, 5):
        #         delta_mags[exposure][ccdid][qid-1] = []

        #         key = 'mag'
        #         refexposure_quadrant = band_path.joinpath("{}_c{}_o_q{}".format(refexposure, str(ccdid).zfill(2), qid))
        #         exposure_quadrant = band_path.joinpath("{}_c{}_o_q{}".format(exposure, str(ccdid).zfill(2), qid))
        #         cat_ref, cat_exp, cat_gaia = match_gaia_catalogs(refexposure_quadrant, exposure_quadrant)

        #         cat_exp['mag'] = -2.5*np.log10(cat_exp['flux'])
        #         cat_exp['emag'] = -1.08*cat_exp['eflux']/cat_exp['flux']

        #         gmag_binned, mag_binned, emag_binned = binplot(cat_gaia['gmag'], cat_exp['mag'], robust=True, data=False, weights=1./cat_exp['emag']**2, bins=np.linspace(min_mag, max_mag, 15))

        #         poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gmag_binned, mag_binned, 0, w=1./emag_binned, full=True)
        #         poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gmag_binned, mag_binned, 0, w=1./emag_binned, full=True)
        #         poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gmag_binned, mag_binned, 0, w=1./emag_binned, full=True)

        #         d_mag = cat_ref['mag'] - cat_exp['mag']
        #         d_mag = d_mag - np.mean(d_mag)
        #         #delta_mags[exposure][ccdid][qid] = [d_mag, cat_gaia['gmag']]

        #         low_mag, high_mag = _bin_dmag(d_mag, cat_gaia['gmag'])
        #         delta = high_mag-low_mag
        #         if delta < vmin:
        #             vmin = delta
        #         if delta > vmax:
        #             vmax = delta

        #         plt.figure(figsize=(7., 4.))
        #         plt.title("{}-{}: {} - CCD={}, Q={}".format(exposure[4:8], exposure[8:10], filtercode, ccdid, qid))
        #         plt.plot(cat_gaia['gmag'], d_mag, '.')
        #         plt.xlabel("$g_\mathrm{Gaia}$ [mag]")
        #         plt.ylabel("$\Delta m$ [mag]")
        #         plt.axhline(y=low_mag, xmin=0., xmax=(low_band-min_mag)/(max_mag-min_mag), color='black')
        #         plt.axhline(y=high_mag, xmin=(high_band-min_mag)/(max_mag-min_mag), xmax=1., color='black')
        #         props = dict(boxstyle='round', facecolor='xkcd:light blue', alpha=1.)
        #         plt.text(16.5, -0.12, "$\Delta m={:.2f}$ mag".format(delta), bbox=props)
        #         plt.axvline(low_band, color='black')
        #         plt.axvline(high_band, color='black')
        #         plt.ylim([-0.2, 0.2])
        #         plt.xlim([min_mag, max_mag])
        #         plt.fill_betweenx([-0.2, 0.2], high_band, 19., color='None', alpha=0.9, edgecolor='black', hatch='/', lw=0.3)
        #         plt.fill_betweenx([-0.2, 0.2], 13., low_band, color='None', alpha=0.9, edgecolor='black', hatch='/', lw=0.3)
        #         plt.grid()
        #         plt.tight_layout()
        #         plt.savefig(exposure_path.joinpath("{}_{}_{}.png".format(exposure, ccdid, qid)), dpi=200.)
        #         # plt.savefig("out/{}_{}_{}.png".format(exposure, ccdid, qid), dpi=200.)
        #         plt.close()

        #         delta_mags[exposure][ccdid][qid-1] = delta

    # cmap = 'coolwarm'
    # def _plot_focal_plane_dmag(exposure, vmin, vmax):
    #     cm = ScalarMappable(cmap=cmap)
    #     cm.set_clim(vmin=vmin, vmax=vmax)

    #     fig = plt.figure(figsize=(5., 6.), constrained_layout=False)
    #     f1, f2 = fig.subfigures(ncols=1, nrows=2, height_ratios=[12., 0.3], wspace=-1., hspace=-1.)
    #     f1.suptitle("\n{}-{}-{}\n$\Delta m$ [mag]".format(exposure[4:8], exposure[8:10], filtercode), fontsize='large')
    #     plot_ztf_focal_plan_values(f1, delta_mags[exposure], vmin=vmin, vmax=vmax, cmap=cmap, scalar=True)
    #     ax2 = f2.add_subplot(clip_on=False)
    #     f2.subplots_adjust(bottom=-5, top=5.)
    #     ax2.set_position((0.2, 0, 0.6, 1))
    #     cb = f1.colorbar(cm, cax=ax2, orientation='horizontal')
    #     ax2.set_clip_on(False)
    #     ax2.minorticks_on()
    #     ax2.tick_params(direction='inout')
    #     plt.savefig(output_path.joinpath("{}-{}_dmag_focal_plane.png".format(exposure, filtercode)), dpi=200., bbox_inches='tight', pad_inches=0.1)
    #     plt.close()

    # vext = max([abs(vmin), abs(vmax)])
    # print(vext)

    # for exposure in exposures:
    #     _plot_focal_plane_dmag(exposure, -vext, vext)
