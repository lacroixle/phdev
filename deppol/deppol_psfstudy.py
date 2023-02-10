#!/usr/bin/env python3

def psfstudy(band_path, ztfname, filtercode, logger, args):
    import pathlib

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar

    from deppol_utils import quadrants_from_band_path
    from utils import get_header_from_quadrant_path, match_gaia_catalogs, plot_ztf_focal_plan_values

    matplotlib.use('Agg')

    output_path = band_path.joinpath("psfstudy")
    output_path.mkdir(exist_ok=True)

    quadrants = quadrants_from_band_path(band_path, logger)

    refexpid = args.refexp
    exposures = list(set([quadrant.name[:28] for quadrant in quadrants]))
    quad2expid = dict([(get_header_from_quadrant_path(band_path.joinpath("{}_c01_o_q1".format(exposure)))['expid'], exposure) for exposure in exposures])
    expid2quad = dict([(k, v) for v, k in zip(quad2expid.values(), quad2expid.keys())])
    print(expid2quad)

    refexposure = expid2quad[refexpid]
    exposures.remove(refexposure)

    delta_mags = {}

    low_band = 15.
    high_band = 16.

    def _bin_dmag(d_mag, gmag):
        low_mask = gmag < low_band
        high_mask = gmag > high_band
        return np.median(d_mag[low_mask]), np.median(d_mag[high_mask])

    vmin = float('inf')
    vmax = -float('inf')

    # exposures = exposures[:1]
    for exposure in exposures:
        delta_mags[exposure] = {}
        exposure_path = output_path.joinpath(exposure)
        exposure_path.mkdir(exist_ok=True)
        for ccdid in range(1, 17):
            delta_mags[exposure][ccdid] = {}
            for qid in range(1, 5):
                print(exposure, ccdid, qid)
                delta_mags[exposure][ccdid][qid-1] = []
                refexposure_quadrant = band_path.joinpath("{}_c{}_o_q{}".format(refexposure, str(ccdid).zfill(2), qid))
                exposure_quadrant = band_path.joinpath("{}_c{}_o_q{}".format(exposure, str(ccdid).zfill(2), qid))
                cat_ref, cat_exp, cat_gaia = match_gaia_catalogs(refexposure_quadrant, exposure_quadrant)

                cat_ref['mag'] = -2.5*np.log10(cat_ref['flux'])
                cat_exp['mag'] = -2.5*np.log10(cat_exp['flux'])

                d_mag = cat_ref['mag'] - cat_exp['mag']
                d_mag = d_mag - np.mean(d_mag)
                #delta_mags[exposure][ccdid][qid] = [d_mag, cat_gaia['gmag']]

                min_mag, max_mag = 13., 19.
                low_mag, high_mag = _bin_dmag(d_mag, cat_gaia['gmag'])
                delta = high_mag-low_mag
                if delta < vmin:
                    vmin = delta
                if delta > vmax:
                    vmax = delta

                plt.figure(figsize=(7., 4.))
                plt.title("{}-{}: {} - CCD={}, Q={}".format(exposure[4:8], exposure[8:10], filtercode, ccdid, qid))
                plt.plot(cat_gaia['gmag'], d_mag, '.')
                plt.xlabel("$g_\mathrm{Gaia}$ [mag]")
                plt.ylabel("$\Delta m$ [mag]")
                plt.axhline(y=low_mag, xmin=0., xmax=(low_band-min_mag)/(max_mag-min_mag), color='black')
                plt.axhline(y=high_mag, xmin=(high_band-min_mag)/(max_mag-min_mag), xmax=1., color='black')
                props = dict(boxstyle='round', facecolor='xkcd:light blue', alpha=1.)
                plt.text(16.5, -0.12, "$\Delta m={:.2f}$ mag".format(delta), bbox=props)
                plt.axvline(low_band, color='black')
                plt.axvline(high_band, color='black')
                plt.ylim([-0.2, 0.2])
                plt.xlim([min_mag, max_mag])
                plt.fill_betweenx([-0.2, 0.2], high_band, 19., color='None', alpha=0.9, edgecolor='black', hatch='/', lw=0.3)
                plt.fill_betweenx([-0.2, 0.2], 13., low_band, color='None', alpha=0.9, edgecolor='black', hatch='/', lw=0.3)
                plt.grid()
                plt.tight_layout()
                plt.savefig(exposure_path.joinpath("{}_{}_{}.png".format(exposure, ccdid, qid)), dpi=200.)
                # plt.savefig("out/{}_{}_{}.png".format(exposure, ccdid, qid), dpi=200.)
                plt.close()

                delta_mags[exposure][ccdid][qid-1] = delta

    cmap = 'coolwarm'
    def _plot_focal_plane_dmag(exposure, vmin, vmax):
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

    vext = max([abs(vmin), abs(vmax)])
    print(vext)

    for exposure in exposures:
        _plot_focal_plane_dmag(exposure, -vext, vext)
