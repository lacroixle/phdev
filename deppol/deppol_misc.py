#!/usr/bin/env python3


def psf_resid(exposure, logger, args):
    from utils import ListTable
    import numpy as np
    import matplotlib.pyplot as plt

    resid_df = ListTable.from_filename(exposure.path.joinpath("psf_resid_tuple.dat")).df
    star_ids = list(set(resid_df['obj']))
    magnification = 2
    star_resid_dict = {}
    pos_list = []
    resid_list = []
    for star_id in star_ids:
        resid_star_df = resid_df.loc[resid_df['obj']==star_id]
        star_offset = np.array([resid_star_df.iloc[0]['xc']-resid_star_df.iloc[0]['ic'], resid_star_df.iloc[0]['yc']-resid_star_df.iloc[0]['jc']])
        pos_list.append(resid_star_df[['i', 'j']].to_numpy()+star_offset)
        resid_list.append((resid_star_df['fpsf']-resid_star_df['fimg']).to_numpy())
        quarter_stamp_size = np.array([resid_df['i'].max(), resid_df['j'].max()])
        # resid = np.empty(2*quarter_stamp_size+1)
        # for idx, [i, j, fpsf, fimg] in resid_star_df[['i', 'j', 'fpsf', 'fimg']].iterrows():
        #     resid[int(i+quarter_stamp_size[0]), int(j+quarter_stamp_size[1])] = fpsf-fimg
        # star_resid_dict[star_id] = {'offset': star_offset, 'resid': resid}

    # resid = np.sum(np.stack([star_resid_dict[star_id]['resid'] for star_id in star_ids]), axis=0)
    pos = magnification*np.concatenate(pos_list)
    resid = np.concatenate(resid_list)
    plt.subplots(figsize=(7., 7.))
    plt.suptitle(exposure.name)
    plt.hist2d(pos[:, 0], pos[:, 1], weights=resid, range=[[-quarter_stamp_size[0], quarter_stamp_size[0]], [-quarter_stamp_size[1], quarter_stamp_size[1]]], bins=30)
    plt.colorbar()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return True


def psf_study(exposure, logger, args):
    import numpy as np
    from saunerie.plottools import binplot
    from yaml import dump
    from numpy.polynomial.polynomial import Polynomial
    import pandas as pd
    from utils import RobustPolynomialFit
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    pd.options.mode.chained_assignment = None

    if not exposure.path.joinpath("psfstars.list").exists():
        return True

    apfl = 'apfl6'
    psf_stars_df = exposure.get_matched_catalog('psfstars')
    aper_stars_df = exposure.get_matched_catalog('aperstars')
    gaia_stars_df = exposure.get_matched_ext_catalog('gaia')

    to_remove = np.any([psf_stars_df['flux'] < 0., aper_stars_df[apfl] < 0., gaia_stars_df['Gmag']>=20.], axis=0)
    psf_stars_df = psf_stars_df.loc[~to_remove]
    aper_stars_df = aper_stars_df.loc[~to_remove]
    gaia_stars_df = gaia_stars_df.loc[~to_remove]

    def _aperflux_to_mag(aper, cat_df):
        cat_df['mag_{}'.format(aper)] = -2.5*np.log10(cat_df[aper])
        cat_df['emag_{}'.format(aper)] = 1.08*cat_df['e{}'.format(aper)]/cat_df[aper]

    psf_stars_df['mag'] = -2.5*np.log10(psf_stars_df['flux'])
    psf_stars_df['emag'] = 1.08*psf_stars_df['eflux']/psf_stars_df['flux']
    _aperflux_to_mag(apfl, aper_stars_df)

    d_mag = (psf_stars_df['mag'] - aper_stars_df['mag_{}'.format(apfl)]).to_numpy()
    d_mag = d_mag - np.nanmean(d_mag)
    ed_mag = np.sqrt(psf_stars_df['emag']**2+aper_stars_df['emag_{}'.format(apfl)]**2).to_numpy()

    d_mag_mask = ~np.isnan(d_mag)
    d_mag = d_mag[d_mag_mask]
    ed_mag = ed_mag[d_mag_mask]

    min_mag, max_mag = 13., 20.

    bins = np.arange(min_mag, max_mag, 0.5)
    plt.subplots(ncols=1, nrows=1, figsize=(8., 5.))
    plt.title("{} - PSF/aper magnitude plot".format(exposure.name))
    plt.errorbar(gaia_stars_df.iloc[d_mag_mask]['Gmag'], d_mag, yerr=ed_mag, ls="", marker='.')
    gmag_binned, d_mag_binned, ed_mag_binned = binplot(gaia_stars_df.iloc[d_mag_mask]['Gmag'].to_numpy(), d_mag, robust=True, data=False, scale=True, weights=1./ed_mag, bins=bins, color='red', lw=2., zorder=15)

    bin_mask = (ed_mag_binned > 0.)
    gmag_binned = gmag_binned[bin_mask]
    d_mag_binned = np.array(d_mag_binned)[bin_mask]
    ed_mag_binned = ed_mag_binned[bin_mask]

    def _poly_fit(degree):
        return RobustPolynomialFit(gmag_binned, d_mag_binned, degree, dy=ed_mag_binned, just_chi2=True)

    # poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 0, w=1./ed_mag_binned, full=True)
    # poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 1, w=1./ed_mag_binned, full=True)
    # poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 2, w=1./ed_mag_binned, full=True)
    try:
        poly0, ([poly0_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 0, w=1./ed_mag_binned, full=True)
        poly1, ([poly1_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 1, w=1./ed_mag_binned, full=True)
        poly2, ([poly2_chi2], _, _, _) = Polynomial.fit(gmag_binned, d_mag_binned, 2, w=1./ed_mag_binned, full=True)
    except Exception as e:
        print("Can't fit polynomial for quadrant {}!".format(exposure.name))
        print(e)
        return False

    polyfit0_chi2 = poly0_chi2/(len(gmag_binned)-1)
    polyfit1_chi2 = poly1_chi2/(len(gmag_binned)-2)
    polyfit2_chi2 = poly2_chi2/(len(gmag_binned)-3)

    plt.plot(gmag_binned, poly0(gmag_binned), label="Order 0 - $\\chi^2={{{}}}$".format(polyfit0_chi2), lw=4.)
    plt.plot(gmag_binned, poly1(gmag_binned), label="Order 1 - $\\chi^2={{{}}}$".format(polyfit1_chi2), lw=4.)
    plt.plot(gmag_binned, poly2(gmag_binned), label="Order 2 - $\\chi^2={{{}}}$".format(polyfit2_chi2), lw=4.)
    plt.legend()
    plt.xlabel("$G$ [Gaia mag]")
    plt.ylabel("$m_\mathrm{PSF}-m_\mathrm{aper}$ [mag]")
    plt.ylim([-0.6, 0.6])
    plt.grid()


    # poly0_chi2 = _poly_fit(0)
    # poly1_chi2 = _poly_fit(1)
    # poly2_chi2 = _poly_fit(2)

    # print(poly0_chi2, polyfit0_chi2)
    # print(poly1_chi2, polyfit1_chi2)
    # print(poly2_chi2, polyfit2_chi2)
    # print("="*100)

    plt.savefig(exposure.path.joinpath("psfaperfit.png"), dpi=250.)
    plt.close()
    with open(exposure.path.joinpath("psfskewness.yaml"), 'w') as f:
        dump({'poly0_chi2': polyfit0_chi2.item(),
              'poly1_chi2': polyfit1_chi2.item(),
              'poly2_chi2': polyfit2_chi2.item()}, f)

    return True


def psf_study_reduce(lightcurve, logger, args):
    import pandas as pd
    from yaml import load, Loader

    exposures = lightcurve.get_exposures(files_to_check='psfskewness.yaml')
    headers = lightcurve.extract_exposure_catalog(files_to_check='psfskewness.yaml')

    psfskewness_list = []
    for exposure in exposures:
        with open(exposure.path.joinpath("psfskewness.yaml"), 'r') as f:
            psfskewness = load(f, Loader=Loader)

        header = headers.loc[exposure.name]

        d = {'quadrant': exposure.name,
             'poly0_chi2': psfskewness['poly0_chi2'],
             'poly1_chi2': psfskewness['poly1_chi2'],
             'poly2_chi2': psfskewness['poly2_chi2'],
             'airmass': float(header['airmass']),
             'mjd': float(header['mjd']),
             'seeing': float(header['seeing']),
             'gfseeing': float(header['gfseeing']),
             'field': header['field'],
             'ccdid': header['ccdid'],
             'qid': header['qid'],
             'rcid': header['rcid'],
             'filtercode': header['filtercode'],
             'skylev': header['skylev'],
             'gain': header['gain']}

        psfskewness_list.append(d)

    df = pd.DataFrame(psfskewness_list)
    df.to_parquet(lightcurve.path.joinpath("psfskewness.parquet"))

    return True

def retrieve_catalogs(lightcurve, logger, args):
    from ztfquery.fields import get_rcid_centroid, FIELDSNAMES
    from ztfimg.catalog import download_vizier_catalog
    import pandas as pd
    import numpy as np
    from croaks.match import NearestNeighAssoc
    import matplotlib
    import matplotlib.pyplot as plt

    from utils import ps1_cat_remove_bad, get_ubercal_catalog_in_cone, write_ds9_reg_circles

    matplotlib.use('Agg')

    # Retrieve Gaia and PS1 catalogs
    exposures = lightcurve.get_exposures()

    # If all exposures are in the primary or secondary field grids, their positions are already known and headers are not needed
    # If a footprint file is provided, use those
    if args.footprints:
        footprints_df = pd.read_csv(args.footprints)
        footprints_df = footprints_df.loc[footprints_df['year']==int(lightcurve.name)].loc[footprints_df['filtercode']==lightcurve.filterid]
        field_rcid_pairs = [("{}-{}-{}".format(args.footprints.stem, footprint.year, footprint.filtercode), i) for i, footprint in footprints_df.iterrows()]
        centroids = [(footprint['ra'], footprint['dec']) for i, footprint in footprints_df.iterrows()]
        radius = [footprint['radius'] for i, footprint in footprints_df.iterrows()]
    else:
        field_rcid_pairs = list(set([(exposure.field, exposure.rcid) for exposure in exposures]))
        if all([(field_rcid_pair[0] in FIELDSNAMES) for field_rcid_pair in field_rcid_pairs]):
            centroids = [get_rcid_centroid(rcid, field) for field, rcid in field_rcid_pairs]
            radius = [0.6]*len(field_rcid_pairs)
        else:
            logger.info("Not all fields are in the primary or secondary grid and no footprint file is provided.")
            logger.info("Retrieving catalogs for each individual quadrant... this may take some time as exposures need to be retrieved.")
            [exposure.retrieve_exposure(force_rewrite=False) for exposure in exposures]
            logger.info("All exposures retrieved.")
            field_rcid_pairs = [(exposure.field, exposure.rcid) for exposure in exposures]
            centroids = [exposure.center() for exposure in exposures]
            radius = [0.6]*len(exposures)


    logger.info("Retrieving catalogs for (fieldid, rcid) pairs: {}".format(field_rcid_pairs))

    name_to_objid = {'gaia': 'Source', 'ps1': 'objID'}

    def _download_catalog(name, centroids, radius):
        catalogs = []
        catalog_df = None
        for centroid, r, field_rcid_pair in zip(centroids, radius, field_rcid_pairs):
            logger.info("{}-{}".format(*field_rcid_pair))

            # First check if catalog already exists in cache
            download_ok = False
            if args.ext_catalog_cache:
                cached_catalog_filename = args.ext_catalog_cache.joinpath("{name}/{name}-{field}-{rcid}.parquet".format(name=name, field=field_rcid_pair[0], rcid=field_rcid_pair[1]))
                if cached_catalog_filename.exists():
                    df = pd.read_parquet(cached_catalog_filename)
                    download_ok = True

            if not download_ok:
                # If not, download it. If a cache path is setup, save it there for future use
                for i in range(5):
                    df = None
                    try:
                        df = download_vizier_catalog(name, centroid, radius=r)
                    except OSError as e:
                        logger.error(e)

                    if df is not None and len(df) > 0:
                        download_ok = True
                        break

                if not download_ok:
                    logger.error("Could not download catalog {}-{}-{} after 5 retries!".format(name, field_rcid_pair[0], field_rcid_pair[1]))
                    continue

                if args.ext_catalog_cache:
                    args.ext_catalog_cache.joinpath(name).mkdir(exist_ok=True)
                    df.to_parquet(cached_catalog_filename)

            if catalog_df is None:
                catalog_df = df
            else:
                catalog_df = pd.concat([catalog_df, df]).drop_duplicates(ignore_index=True, subset=name_to_objid[name])

            logger.info("Catalog size={}".format(len(catalog_df)))

        return catalog_df

    def _get_catalog(name, centroids, radius):
        catalog_path = lightcurve.ext_catalogs_path.joinpath("{}_full.parquet".format(name))
        logger.info("Getting catalog {}... ".format(name))
        if not catalog_path.exists() or args.recompute_ext_catalogs:
            catalog_df = _download_catalog(name, centroids, radius)
            catalog_path.parents[0].mkdir(exist_ok=True)

            catalog_df.dropna(subset=['ra', 'dec'], inplace=True)

            # Remove low detection counts PS1 objects, variable objects, possible non stars objects
            if name == 'ps1':
                catalog_df = catalog_df.loc[catalog_df['Nd']>=20].reset_index(drop=True)
                catalog_df = ps1_cat_remove_bad(catalog_df)
                # catalog_df = catalog_df.loc[catalog_df['gmag']-catalog_df['rmag']<=1.5]
                # catalog_df = catalog_df.loc[catalog_df['gmag']-catalog_df['rmag']>=0.]
                catalog_df.dropna(subset=['gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag'], inplace=True)

            if name == 'gaia':
                # Change proper motion units
                catalog_df['pmRA'] = catalog_df['pmRA']/np.cos(np.deg2rad(catalog_df['dec']))/1000./3600./365.25 # TODO: check if dec should be J2000 or something else
                catalog_df['pmDE'] = catalog_df['pmDE']/1000./3600./365.25

                # catalog_df = catalog_df.loc[catalog_df['Gmag'] >= 10.]
                # catalog_df = catalog_df.loc[catalog_df['Gmag'] <= 23.]
                # catalog_df = catalog_df.loc[catalog_df['BP-RP'] <= 2.]
                # catalog_df = catalog_df.loc[catalog_df['BP-RP'] >= -0.5]
                catalog_df.dropna(subset=['Gmag', 'e_Gmag', 'BPmag', 'e_BPmag', 'RPmag', 'e_RPmag'], inplace=True)

            logger.info("Saving catalog into {}".format(catalog_path))
            catalog_df.to_parquet(catalog_path)
        else:
            logger.info("Found: {}".format(catalog_path))
            catalog_df = pd.read_parquet(catalog_path)

        return catalog_df

    def _get_ubercal_catalog(name, centroids, radius):
        catalog_path = lightcurve.ext_catalogs_path.joinpath("ubercal_{}_full.parquet".format(name))
        logger.info("Getting Ubercal catalog {}...".format(name))
        if not catalog_path.exists() or args.recompute_ext_catalogs:
            catalog_df = None
            for centroid, r in zip(centroids, radius):
                df = get_ubercal_catalog_in_cone(name, args.ubercal_config_path, centroid[0], centroid[1], r)

                if catalog_df is None:
                    catalog_df = df
                else:
                    catalog_df = pd.concat([catalog_df, df])
                    catalog_df = catalog_df[~catalog_df.index.duplicated(keep='first')]

            logger.info("Saving catalog into {}".format(catalog_path))
            catalog_df.to_parquet(catalog_path)
        else:
            logger.info("Found: {}".format(catalog_path))
            catalog_df = pd.read_parquet(catalog_path)

        return catalog_df

    if args.lc_folder is not None:
        sn_parameters = pd.read_hdf(args.lc_folder.joinpath("{}.hd5".format(lightcurve.name)), key='sn_info')

    def _plot_catalog_coverage(cat_df, name):
        plt.subplots(figsize=(6., 6.))
        plt.suptitle("Coverage for catalog {}".format(name))
        plt.plot(cat_df['ra'].to_numpy(), cat_df['dec'].to_numpy(), '.', label="Catalog stars")
        if args.lc_folder is not None:
            plt.plot(sn_parameters['sn_ra'], sn_parameters['sn_dec'], 'x', label="SN")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(lightcurve.ext_catalogs_path.joinpath("catalog_coverage_{}.png".format(name)), dpi=300.)
        plt.close()

    logger.info("Retrieving external catalogs")

    gaia_df = _get_catalog('gaia', centroids, radius)
    ps1_df = _get_catalog('ps1', centroids, radius)

    _plot_catalog_coverage(gaia_df, 'gaia_full')
    _plot_catalog_coverage(ps1_df, 'ps1_full')

    write_ds9_reg_circles(lightcurve.ext_catalogs_path.joinpath("gaia_catalog.reg"), gaia_df[['ra', 'dec']].to_numpy(), [10.]*len(gaia_df))
    write_ds9_reg_circles(lightcurve.ext_catalogs_path.joinpath("ps1_catalog.reg"), ps1_df[['ra', 'dec']].to_numpy(), [8]*len(ps1_df))

    logger.info("Matching Gaia and PS1 catalogs")
    # For both catalogs, radec are in J2000 epoch, so no need to account for space motion
    assoc = NearestNeighAssoc(first=[gaia_df['ra'].to_numpy(), gaia_df['dec'].to_numpy()], radius = 2./60./60.)
    i = assoc.match(ps1_df['ra'].to_numpy(), ps1_df['dec'].to_numpy())

    gaia_df = gaia_df.iloc[i[i>=0]].reset_index(drop=True)
    ps1_df = ps1_df.iloc[i>=0].reset_index(drop=True)

    write_ds9_reg_circles(lightcurve.ext_catalogs_path.joinpath("joined_catalog.reg"), gaia_df[['ra', 'dec']].to_numpy(), [10.]*len(gaia_df))

    if args.ubercal_config_path:
        # logger.info("Retrieving self Ubercal")
        # ubercal_self_df = _get_ubercal_catalog('self', centroids, radius)
        # logger.info("Found {} stars".format(len(ubercal_self_df)))
        # _plot_catalog_coverage(ubercal_self_df, 'ubercal_self_full')

        # logger.info("Retrieving PS1 Ubercal")
        # ubercal_ps1_df = _get_ubercal_catalog('ps1', centroids, radius)
        # logger.info("Found {} stars".format(len(ubercal_ps1_df)))
        # _plot_catalog_coverage(ubercal_ps1_df, 'ubercal_ps1_full')

        logger.info("Retrieving fluxcatalog Ubercal")
        ubercal_fluxcatalog_df = _get_ubercal_catalog('fluxcatalog', centroids, radius)
        logger.info("Found {} stars".format(len(ubercal_fluxcatalog_df)))
        _plot_catalog_coverage(ubercal_fluxcatalog_df, 'ubercal_fluxcatalog')

        # ubercal_fluxcatalog_df = ubercal_fluxcatalog_df.loc[ubercal_fluxcatalog_df['is_calibrator'].apply(bool)]

        # common_gaiaids = list(set(set(ubercal_self_df.index) & set(ubercal_ps1_df.index) & set(gaia_df['Source'])))
        # logger.info("Keeping {} stars in common in all catalogs".format(len(common_gaiaids)))
        # gaiaid_mask = gaia_df['Source'].apply(lambda x: x in common_gaiaids).tolist()
        # gaia_df = gaia_df.loc[gaiaid_mask]
        # ps1_df = ps1_df.loc[gaiaid_mask]
        # ps1_df = ps1_df.set_index(gaia_df['Source']).loc[common_gaiaids].reset_index(drop=True)
        # gaia_df = gaia_df.set_index('Source').loc[common_gaiaids].reset_index()
        # ubercal_self_df = ubercal_self_df.filter(items=common_gaiaids, axis=0).reset_index()
        # ubercal_ps1_df = ubercal_ps1_df.filter(items=common_gaiaids, axis=0).reset_index()

        # Self Ubercal color-color plot
        plt.subplots(figsize=(6., 6.))
        plt.suptitle("Fluxcatalog Ubercal color-color plot")
        plt.scatter((ubercal_fluxcatalog_df['zgmag']-ubercal_fluxcatalog_df['zrmag']).to_numpy(), (ubercal_fluxcatalog_df['zrmag']-ubercal_fluxcatalog_df['zimag']).to_numpy(), c=ubercal_fluxcatalog_df['zgmag'], s=1.)
        plt.xlabel("$g_\mathrm{Ubercal}-r_\mathrm{Ubercal}$ [mag]")
        plt.ylabel("$r_\mathrm{Ubercal}-i_\mathrm{Ubercal}$ [mag]")
        plt.axis('equal')
        plt.grid()
        plt.colorbar(label="$g_\mathrm{Ubercal}$ [mag]")
        plt.tight_layout()
        plt.savefig(lightcurve.ext_catalogs_path.joinpath("fluxcatalog_ubercal_color_color.png"), dpi=300.)
        plt.close()

        logger.info("Saving Ubercal catalogs")
        ubercal_fluxcatalog_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("ubercal_fluxcatalog.parquet"))
        # ubercal_self_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("ubercal_self.parquet"))
        # ubercal_ps1_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("ubercal_ps1.parquet"))

    _plot_catalog_coverage(gaia_df, 'matched')
    write_ds9_reg_circles("out.reg", gaia_df[['ra', 'dec']].to_numpy(), [10]*len(gaia_df))

    # PS1 color-color plot
    plt.subplots(figsize=(6., 6.))
    plt.suptitle("PS1 color-color plot")
    plt.scatter((ps1_df['gmag']-ps1_df['rmag']).to_numpy(), (ps1_df['rmag']-ps1_df['imag']).to_numpy(), c=ps1_df['gmag'], s=1.)
    plt.xlabel("$g_\mathrm{PS1}-r_\mathrm{PS1}$ [mag]")
    plt.ylabel("$r_\mathrm{PS1}-i_\mathrm{PS1}$ [mag]")
    plt.axis('equal')
    plt.grid()
    plt.colorbar(label="$g_\mathrm{PS1}$ [mag]")
    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ps1_color_color.png"), dpi=300.)
    plt.close()

    logger.info("Saving Gaia/PS1 matched catalogs")
    gaia_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("gaia.parquet"))
    ps1_df.to_parquet(lightcurve.ext_catalogs_path.joinpath("ps1.parquet"))

    return True


def plot_catalogs(lightcurve, logger, args):
    import matplotlib.pyplot as plt
    import numpy as np

    from saunerie.plottools import binplot
    from croaks.match import NearestNeighAssoc
    from utils import mag2extcatmag, emag2extcatemag

    gaia_df = lightcurve.get_ext_catalog('gaia')
    ps1_df = lightcurve.get_ext_catalog('ps1')
    ubercal_df = lightcurve.get_ext_catalog('ubercal_fluxcatalog')

    assoc = NearestNeighAssoc(first=[ps1_df['ra'].to_numpy(), ps1_df['dec'].to_numpy()], radius = 1./60./60.)
    i = assoc.match(ubercal_df['ra'].to_numpy(), ubercal_df['dec'].to_numpy())

    ps1_df = ps1_df.iloc[i[i>=0]].reset_index(drop=True)
    ubercal_df = ubercal_df.iloc[i>=0].reset_index(drop=True)

    # Plot Ubercal RMS
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0.}, sharex=True)
    plt.suptitle("Ubercal catalog - Star RMS")
    plt.subplot(3, 1, 1)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$g$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$r$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$i$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_rms.png"), dpi=200.)
    plt.close()

    # Plot Ubercal RMS zoom
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0.}, sharex=True)
    plt.suptitle("Ubercal catalog - Star RMS - Zoomed in")
    plt.subplot(3, 1, 1)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$g$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$r$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(ps1_df['gmag'], ubercal_df['zgrms'], '.')
    plt.ylabel("RMS - ZTF-$i$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_rms_zoom.png"), dpi=200.)
    plt.close()

    # Plot Ubercal RMS
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0.}, sharex=True)
    plt.suptitle("Ubercal catalog - Mag std")
    plt.subplot(3, 1, 1)
    plt.plot(ps1_df['gmag'], ubercal_df['ezgmag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$g$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(ps1_df['gmag'], ubercal_df['ezrmag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$r$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(ps1_df['gmag'], ubercal_df['ezimag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$i$ [mag]")
    plt.ylim(0., 0.5)
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_err.png"), dpi=200.)
    plt.close()

    # Plot Ubercal RMS zoom
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0.}, sharex=True)
    plt.suptitle("Ubercal catalog - Star RMS - Zoomed in")
    plt.subplot(3, 1, 1)
    plt.plot(ps1_df['gmag'], ubercal_df['ezgmag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$g$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(ps1_df['gmag'], ubercal_df['ezrmag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$r$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(ps1_df['gmag'], ubercal_df['ezimag'], '.')
    plt.ylabel("$\sigma_m$ - ZTF-$i$ [mag]")
    plt.ylim(0., 0.1)
    plt.grid()

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_err_zoom.png"), dpi=200.)
    plt.close()
    # First compare PS1 catalog with Ubercal's PS1 catalog
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0}, sharex=True)
    plt.suptitle("Comparing PS1 catalogs with Ubercal's PS1 catalogs")
    plt.subplot(3, 1, 1)
    plt.scatter(ps1_df['gmag'], ps1_df['gmag']-ubercal_df['g_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 2)
    plt.scatter(ps1_df['gmag'], ps1_df['rmag']-ubercal_df['r_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 3)
    plt.scatter(ps1_df['gmag'], ps1_df['imag']-ubercal_df['i_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.xlabel("$m_g^\mathrm{PS1}$ [mag]")
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_ps1_ps1_comparison.png"), dpi=200.)
    plt.close()

    # Same, but zoomed in
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0}, sharex=True)
    plt.suptitle("Comparing PS1 catalogs with Ubercal's PS1 catalogs")
    plt.subplot(3, 1, 1)
    plt.scatter(ps1_df['gmag'], ps1_df['gmag']-ubercal_df['g_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.01, 0.01)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 2)
    plt.scatter(ps1_df['gmag'], ps1_df['rmag']-ubercal_df['r_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.01, 0.01)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 3)
    plt.scatter(ps1_df['gmag'], ps1_df['imag']-ubercal_df['i_mag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.01, 0.01)
    plt.xlabel("$m_g^\mathrm{PS1}$ [mag]")
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_ps1_ps1_comparison_zoom.png"), dpi=200.)
    plt.close()


    # Compare PS1 catalog with Ubercal
    plt.subplots(nrows=3, ncols=1, figsize=(15., 9.), gridspec_kw={'hspace': 0}, sharex=True)
    plt.suptitle("Comparing PS1 catalogs with Ubercal's catalogs")
    plt.subplot(3, 1, 1)
    plt.scatter(ps1_df['gmag'], ps1_df['gmag']-ubercal_df['zgmag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 2)
    plt.scatter(ps1_df['gmag'], ps1_df['rmag']-ubercal_df['zrmag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.subplot(3, 1, 3)
    plt.scatter(ps1_df['gmag'], ps1_df['imag']-ubercal_df['zimag'], c=(ps1_df['gmag']-ps1_df['imag']), s=0.5)
    plt.grid()
    plt.ylim(-0.2, 0.2)
    plt.xlabel("$m_g^\mathrm{PS1}$ [mag]")
    plt.ylabel("$m_g^\mathrm{PS1}-m_g^\mathrm{Ubercal}$ [mag]")

    plt.tight_layout()
    plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_ps1.png"), dpi=200.)
    plt.close()

    # Compare PS1 catalog with Ubercal - binplots
    def _ubercal_ps1_binplot(filtercode):
        mag_ps1 = ps1_df[mag2extcatmag['ps1'][filtercode]].to_numpy()
        mag_ubercal = ubercal_df[mag2extcatmag['ubercal_fluxcatalog'][filtercode]].to_numpy()
        emag_ps1 = ps1_df[emag2extcatemag['ps1'][filtercode]].to_numpy()
        emag_ubercal = ubercal_df[emag2extcatemag['ubercal_fluxcatalog'][filtercode]].to_numpy()
        weights = 1./np.sqrt(emag_ps1**2+emag_ubercal**2)

        plt.subplots(nrows=3, ncols=1, figsize=(12., 8.), sharex=True, gridspec_kw={'hspace': 0.})
        plt.suptitle("Comparing PS1 and Ubercal catalogs - ${}$ band".format(filtercode[1]))
        plt.subplot(3, 1, 1)
        mag_binned, d_mag_binned, ed_mag_binned = binplot(mag_ps1, mag_ps1-mag_ubercal, weights=weights, robust=True, data=False, scale=False)
        plt.scatter(mag_ps1, mag_ps1-mag_ubercal, c=(ps1_df['gmag']-ps1_df['imag']).to_numpy(), s=0.8)
        plt.ylim(-0.4, 0.4)
        plt.ylabel("$m_{{{filtercode}}}^\mathrm{{PS1}}-m_{{{filtercode}}}^\mathrm{{Ubercal}}$ [mag]".format(filtercode=filtercode[1]))
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(mag_binned, d_mag_binned)
        plt.ylabel("$\langle m_{{{filtercode}}}^\mathrm{{PS1}}-m_{{{filtercode}}}^\mathrm{{Ubercal}} \\rangle$ [mag]".format(filtercode=filtercode[1]))
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(mag_binned, ed_mag_binned)
        plt.xlabel("$m_{{{filtercode}}}^\mathrm{{PS1}}$ [mag]".format(filtercode=filtercode[1]))
        plt.ylabel("$\sigma_{{m_{{{filtercode}}}^\mathrm{{PS1}}-m_{{{filtercode}}}^\mathrm{{Ubercal}}}}$ [mag]".format(filtercode=filtercode[1]))
        plt.grid()

        plt.tight_layout()
        plt.savefig(lightcurve.ext_catalogs_path.joinpath("ubercal_ps1_{}.png".format(filtercode)), dpi=200.)
        plt.close()

    _ubercal_ps1_binplot('zg')
    _ubercal_ps1_binplot('zr')
    _ubercal_ps1_binplot('zi')

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

    if not exposure.path.joinpath("psfstars.list").exists():
        logger.error("makepsf was not successful, no psfstars catalog!")
        return False

    psf_stars_df = exposure.get_catalog("psfstars.list").df
    aper_stars_df = exposure.get_catalog("standalone_stars.list").df

    wcs = exposure.wcs
    mjd = exposure.mjd

    exposure_centroid = exposure.center()
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
    gaia_stars_inside = wcs.footprint_contains(gaia_stars_skycoords)
    gaia_stars_skycoords = gaia_stars_skycoords[gaia_stars_inside]
    gaia_stars_df = gaia_stars_df.iloc[gaia_stars_inside]

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
        hdfstore.put('ext_cat_inside', pd.Series(gaia_stars_inside))

    return True


def clean(lightcurve, logger, args):
    from shutil import rmtree

    def _clean(exposure, logger, args):
        # We want to delete all files in order to get back to the prepare_deppol stage
        files_to_keep = ["elixir.fits", "dead.fits.gz", ".dbstuff"]
        files_to_delete = list(filter(lambda f: f.name not in files_to_keep, list(exposure.path.glob("*"))))

        for file_to_delete in files_to_delete:
                file_to_delete.unlink()

        return True
    exposures = lightcurve.get_exposures(ignore_noprocess=True)
    [_clean(exposure, logger, args) for exposure in exposures]

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

    chi2_df = pd.read_csv(lightcurve.astrometry_path.joinpath("tp2px_chi2.csv"), index_col=0)

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
    calibrated_path = exposure.path.joinpath("calibrated.fits")
    weight_path= exposure.path.joinpath("weight.fz")
    satur_path = exposure.path.joinpath("satur.fits.gz")

    def _delete(path):
        if path.exists():
            logger.info("{} exists and will be deleted.".format(path.name))
        else:
            logger.info("{} does not exists.".format(path.name))

    _delete(calibrated_path)
    _delete(weight_path)
    _delete(satur_path)

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
            logger.error("Could not find catalog {}!".format(catalog_name))
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


def plot_footprint(lightcurve, logger, args):
    import matplotlib.pyplot as plt
    from utils import sc_array

    exposures = lightcurve.get_exposures()[:2000]

    plt.subplots(figsize=(10., 10.))
    plt.suptitle("Sky footprint for {}-{}".format(lightcurve.name, lightcurve.filterid))
    for exposure in exposures:
        wcs = exposure.wcs
        width, height = wcs.pixel_shape
        top_left = [0., height]
        top_right = [width, height]
        bottom_left = [0., 0.]
        bottom_right = [width, 0]

        tl_radec = sc_array(wcs.pixel_to_world(*top_left))
        tr_radec = sc_array(wcs.pixel_to_world(*top_right))
        bl_radec = sc_array(wcs.pixel_to_world(*bottom_left))
        br_radec = sc_array(wcs.pixel_to_world(*bottom_right))
        plt.plot([tl_radec[0], tr_radec[0], br_radec[0], bl_radec[0], tl_radec[0]], [tl_radec[1], tr_radec[1], br_radec[1], bl_radec[1], tl_radec[1]], color='grey')

    if lightcurve.ext_catalogs_path.joinpath("gaia_full.parquet").exists():
        gaia_stars_df = lightcurve.get_ext_catalog('gaia', matched=False)
        plt.plot(gaia_stars_df['ra'].to_numpy(), gaia_stars_df['dec'], ',')

    if lightcurve.ext_catalogs_path.joinpath("ps1_full.parquet").exists():
        ps1_stars_df = lightcurve.get_ext_catalog('ps1', matched=False)
        plt.plot(ps1_stars_df['ra'].to_numpy(), ps1_stars_df['dec'].to_numpy(), ',')

    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$\\delta$ [deg]")
    plt.tight_layout()
    plt.savefig(lightcurve.path.joinpath("footprint.png"), dpi=300.)
    plt.close()

def concat_catalogs(lightcurve, logger, args):
    import pandas as pd
    from astropy import log

    log.setLevel('ERROR')
    logger.info("Retrieving exposures")
    exposures = lightcurve.get_exposures(files_to_check='match_catalogs.success')
    logger.info("Retrieving headers")
    headers = lightcurve.extract_exposure_catalog(files_to_check='match_catalogs.success')

    def _add_prefix_to_column(df, prefix):
        column_names = dict([(column, prefix+column) for column in df.columns])
        df.rename(columns=column_names, inplace=True)

    def _extract_quadrant(ccdid, qid):
        cat_dfs = []
        for exposure in exposures:
            if exposure.ccdid == ccdid and exposure.qid == qid:
                print(".", flush=True, end="")
                aperstars_df = exposure.get_matched_catalog('aperstars')
                psfstars_df = exposure.get_matched_catalog('psfstars')
                gaia_df = exposure.get_matched_ext_catalog('gaia')
                ps1_df = exposure.get_matched_ext_catalog('ps1')
                _add_prefix_to_column(aperstars_df, 'aper_')
                _add_prefix_to_column(psfstars_df, 'psf_')
                _add_prefix_to_column(gaia_df, 'gaia_')
                _add_prefix_to_column(ps1_df, 'ps1_') #
                cat_df = pd.concat([aperstars_df, psfstars_df, gaia_df, ps1_df], axis='columns')
                # cat_df = pd.concat([aperstars_df, psfstars_df, gaia_df], axis='columns')
                cat_df.insert(0, 'quadrant', exposure.name)
                for column in headers.columns:
                    cat_df[column] = headers.at[exposure.name, column]
                cat_dfs.append(cat_df)

        # print("")               #
        if len(cat_dfs) > 0:
            return pd.concat(cat_dfs)
        else:
            return None

    logger.info("Extracting star catalogs")
    lightcurve.path.joinpath("measures").mkdir(exist_ok=True)
    for ccdid in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        for qid in [1, 2, 3, 4]:
            filename = lightcurve.path.joinpath("measures/measures_{}-{}-c{}-q{}.parquet".format(lightcurve.name, lightcurve.filterid, str(ccdid).zfill(2), str(qid)))
            if not filename.exists():
                logger.info("ccdid={}, qid={}".format(ccdid, qid))
                df = _extract_quadrant(ccdid, qid)
                if df is not None:
                    logger.info("Found {} measures".format(len(df)))
                    df.to_parquet(filename)
                else:
                    logger.info("Found no measure")

    return True


def plot_focalplane_stars(lightcurve, logger, args):
    import matplotlib.pyplot as plt
    import pandas as pd
    from pyloka import pix2tp
    import random

    stars_list = []
    exposures = lightcurve.get_exposures()
    for exposure in exposures:
        stars_df = exposure.get_matched_catalog('aperstars')
        foc_x, foc_y = pix2tp(stars_df['x'].to_numpy(), stars_df['y'].to_numpy(), str(args.wd.joinpath("{}/{}/{}/calibrated.fits".format(lightcurve.name, lightcurve.filterid, exposure.name))).encode('utf8'))
        stars_df['foc_x'] = foc_x
        stars_df['foc_y'] = foc_y
        stars_list.append(stars_df)

    stars_df = pd.concat(stars_list)

    plt.subplots(figsize=(10., 10.))
    plt.plot(stars_df['foc_x'].to_numpy(), stars_df['foc_y'].to_numpy(), ',')
    plt.savefig("output.png")
    plt.close()

    return True
