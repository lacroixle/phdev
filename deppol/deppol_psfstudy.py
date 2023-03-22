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
    from utils import get_header_from_quadrant_path, plot_ztf_focal_plan_values, ListTable, match_by_gaia_catalogs, quadrant_name_explode, match_to_gaia

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

    def _process_quadrant(quadrant_path, cat1, cat2):
        def _cat_to_name(cat):
            if cat == 'flux':
                return "PSF"
            else:
                return "Aperture-" + cat[-1]

        def _is_aperture(cat):
            return 'apfl' in cat

        def _get_cat_from_name(cat, refexp=None):
            path = quadrant_path
            if refexp:
                path = quadrant_path.with_name(refexp + quadrant_path.name[28:])

            if cat == 'flux':
                with pd.HDFStore(path.joinpath("matched_stars.hd5")) as hdfstore:
                    cat_df = hdfstore.get('/matched_stars')
            else:
                cat_df = ListTable.from_filename(path.joinpath("standalone_stars.list")).df

            return cat_df

        def _get_gaia_cat(refexp=None):
            path = quadrant_path
            if refexp:
                path = quadrant_path.with_name(refexp + quadrant_path.name[28:])

            with pd.HDFStore(path.joinpath("matched_stars.hd5")) as hdfstore:
                cat_gaia_df = hdfstore.get('/matched_gaia_stars')

            return cat_gaia_df

        if not _quadrant_good(quadrant_path):
            return

        try:
            refexp = None
            if cat1[:3] == 'ref':
                refexp = args.refexp
                cat1 = cat1[3:]

            quadrant_header = get_header_from_quadrant_path(quadrant_path)
            expid = int(quadrant_header['expid'])
            mjd = float(quadrant_header['obsmjd'])
            ccdid = int(quadrant_header['ccdid'])
            qid = int(quadrant_header['qid'])
            skylev = float(quadrant_header['sexsky'])
            moonillf = float(quadrant_header['moonillf'])
            seeing = float(quadrant_header['seeing'])
            airmass = float(quadrant_header['airmass'])

            year, month, day, _, _, ccdid, qid = quadrant_name_explode(quadrant_path.name)

            exposure_output_path = output_path.joinpath("{}-{}-{}-{}".format(year, month, day, expid))
            exposure_output_path.mkdir(exist_ok=True)

            # First load catalogs
            cat1_df = _get_cat_from_name(cat1, refexp=refexp)
            cat2_df = _get_cat_from_name(cat2)

            if refexp:
                cat1_gaia_df = _get_gaia_cat(refexp)
                cat2_gaia_df = _get_gaia_cat()
            else:
                cat1_gaia_df = _get_gaia_cat()
                cat2_gaia_df = cat1_gaia_df

            if _is_aperture(cat1):
                cat1_df['gaiaid'] = match_to_gaia(cat1_df, cat1_gaia_df)

            if _is_aperture(cat2):
                cat2_df['gaiaid'] = match_to_gaia(cat2_df, cat2_gaia_df)

            # Match catalogs to Gaia
            # i = match_pixel_space(cat_gaia_df[['x', 'y']].to_records(), cat2_df[['x', 'y']].to_records(), radius=1.)

            # cat2_df = cat2_df.iloc[i[i>=0]].reset_index(drop=True)
            # cat_gaia_df = cat_gaia_df.iloc[i[i>=0]].reset_index(drop=True)
            # cat1_df = cat1_df.iloc[i>=0].reset_index(drop=True)
            print(cat1_df)
            print(cat2_df)
            cat1_df, cat2_df, cat_gaia_df = match_by_gaia_catalogs(cat1_df, cat2_df, cat1_gaia_df, cat2_gaia_df)

            # Convert ADU flux into mag
            cat1_df['mag'] = -2.5*np.log10(cat1_df[cat1])
            cat2_df['mag'] = -2.5*np.log10(cat2_df[cat2])
            cat1_df['emag'] = -1.08*cat1_df['e'+cat1]/cat1_df[cat1]
            cat2_df['emag'] = -1.08*cat2_df['e'+cat2]/cat2_df[cat2]

            if _is_aperture(cat1):
                aper_size_1 = list(set(cat1_df['rad'+cat1[-1]]))[0]

            if _is_aperture(cat2):
                aper_size_2 = list(set(cat2_df['rad'+cat2[-1]]))[0]

            # Compute delta flux between PSF and aperture photometry
            deltamag = (cat1_df['mag'] - cat2_df['mag']).to_numpy()
            deltamag = deltamag - np.mean(deltamag)
            edeltamag = np.sqrt(cat1_df['emag']**2+cat2_df['emag']**2).to_numpy()

            # Bin delta flux
            plt.figure(figsize=(7., 4.))
            gmag_binned, deltamag_binned, edeltamag_binned = binplot(cat_gaia_df['gmag'].to_numpy(), deltamag, robust=True, data=False, scale=True, weights=1./edeltamag**2, bins=np.linspace(min_mag, max_mag, 8), color='black', zorder=15)

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
            name1 = _cat_to_name(cat1)
            name2 = _cat_to_name(cat2)

            if _is_aperture(cat1):
                name1 += " {:.2f} px".format(aper_size_1)

            if _is_aperture(cat2):
                name2 += " {:.2f} px".format(aper_size_2)

            plt.title("{} - {} [mag]\n({}-{}-{}) CCD={} Q={}\nSky level={:.2f}".format(name1, name2, year, month, day, ccdid, qid, skylev))
            plt.grid()
            gmag_space = np.linspace(min_mag, max_mag, 100)
            plt.plot(gmag_space, poly0(gmag_space), ls='-', lw=1.5, color='grey', label="Constant - $\chi^2_\\nu={:.3f}$".format(poly0_chi2), zorder=10)
            plt.plot(gmag_space, poly1(gmag_space), ls='dotted', lw=1.5, color='grey', label="Linear - $\chi^2_\\nu={:.3f}$".format(poly1_chi2), zorder=10)
            plt.plot(gmag_space, poly2(gmag_space), ls='--', lw=1.5, color='grey', label="Quadratic - $\chi^2_\\nu={:.3f}$".format(poly2_chi2), zorder=10)
            plt.errorbar(cat_gaia_df['gmag'].to_numpy(), deltamag, yerr=edeltamag, lw=0., marker='.', markersize=1.5, zorder=0, color='xkcd:light blue')
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
                    'poly2_chi2': poly2_chi2,
                    'seeing': seeing,
                    'airmass': airmass}

        except Exception as e:
            print(quadrant_path)
            print(e)
            print(traceback.format_exc())


    # d = [_process_quadrant(quadrant_path) for quadrant_path in quadrant_paths]

    tasks = [delayed(_process_quadrant)(quadrant_path, 'flux', 'apfl6') for quadrant_path in quadrant_paths]
    results = compute(tasks)

    # pairs = [('flux', 'apfl6'), ('refflux', 'flux')]
    pairs = [('refflux', 'flux'), ('flux', 'apfl6')]
    print("")
    print("Done.")
    results = list(filter(lambda x: x is not None, results[0]))
    df = pd.DataFrame(results)
    df.to_parquet(band_path.joinpath("myquadrants.parquet"))

    cmap = 'coolwarm'
    def _plot_focal_plane(exposure, vmin, vmax, val):
        cm = ScalarMappable(cmap=cmap)
        cm.set_clim(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(5., 6.), constrained_layout=False)
        f1, f2 = fig.subfigures(ncols=1, nrows=2, height_ratios=[12., 0.3], wspace=-1., hspace=-1.)
        f1.suptitle("\n{}-{}-{}\n$\Delta m$ [mag]".format(exposure[4:8], exposure[8:10], filtercode), fontsize='large')
        plot_ztf_focal_plan_values(f1, delta_mags[exposure], vmin=vmin, vmax=vmax, cmap=cmap, scalar=True)
        ax2 = f2.add_subplot(clip_on=False)
        f2.subplots_adjust(bottom=-5, top=5.)
        ax2.set_position((0.2, 0, 0.6, 1))
        cb = f1.colorbar(cm, cax=ax2, orientation='horizontal')
        ax2.set_clip_on(False)
        ax2.minorticks_on()
        ax2.tick_params(direction='inout')
        plt.savefig(output_path.joinpath("{}-{}_dmag_focal_plane.png".format(exposure, filtercode)), dpi=200., bbox_inches='tight', pad_inches=0.1)
        plt.close()

    fp_vals = {'PSF-Aper': {'limits': 0.2, 'val': lambda x: x.deltamag}}
    for exposure in exposures:
        _plot_focal_plane_dmag(exposure, -vext, vext)
