#!/usr/bin/env python3

from deppol_utils import run_and_log


def make_catalog(exposure, logger, args):
    from ztfquery.io import get_file
    from shutil import copyfile
    from utils import get_header_from_quadrant_path
    import pathlib

    logger.info("Retrieving science exposure...")
    try:
        image_path = exposure.retrieve_exposure(ztfin2p3_path=args.ztfin2p3_path)
    except FileNotFoundError as e:
        print(e)
        logger.error(e)
        return False

    logger.info("Found at {}".format(image_path))

    run_and_log(["make_catalog", exposure.path, "-O", "-S"], logger)

    logger.info("Dumping header content")
    exposure.update_exposure_header()

    return exposure.path.joinpath("se.list").exists()


make_catalog_rm = ["low.fits.gz", "miniback.fits", "segmentation.cv.fits", "segmentation.fits"]


def mkcat2(exposure, logger, args):
    from itertools import chain
    import numpy as np
    from scipy.sparse import dok_matrix
    from utils import match_pixel_space
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from utils import contained_in_exposure, sc_array
    import matplotlib.pyplot as plt

    run_and_log(["mkcat2", exposure.path, "-o"], logger)

    if not exposure.path.joinpath("standalone_stars.list").exists():
        return False

    if args.use_gaia_stars:
        # Find standalone stars using Gaia
        logger.info("Using Gaia catalog to identify stars")
        aperse_cat = exposure.get_catalog("aperse.list")
        standalone_stars_cat = exposure.get_catalog("standalone_stars.list")
        gaia_stars_df = exposure.lightcurve.get_ext_catalog('gaia', matched=False)[['Gmag', 'ra', 'dec']].dropna()

        logger.info("Total Gaia stars={}".format(len(gaia_stars_df)))
        # Remove Gaia stars outside of the exposure
        wcs = exposure.wcs
        gaia_stars_skycoords = SkyCoord(ra=gaia_stars_df['ra'].to_numpy(), dec=gaia_stars_df['dec'].to_numpy(), unit='deg')
        gaia_stars_inside = wcs.footprint_contains(gaia_stars_skycoords)
        inside = sum(gaia_stars_inside)
        gaia_stars_inside = contained_in_exposure(gaia_stars_skycoords, wcs, return_mask=True)
        if np.sum(gaia_stars_inside) == 0:
            print(exposure.name)
            plt.plot(gaia_stars_df['ra'].to_numpy(), gaia_stars_df['dec'], ',')

            width, height = wcs.pixel_shape
            top_left = [0., height]
            top_right = [width, height]
            bottom_left = [0., 0.]
            bottom_right = [width, 0]

            tl_radec = sc_array(wcs.pixel_to_world(*top_left))
            tr_radec = sc_array(wcs.pixel_to_world(*top_right))
            bl_radec = sc_array(wcs.pixel_to_world(*bottom_left))
            br_radec = sc_array(wcs.pixel_to_world(*bottom_right))
            plt.plot([tl_radec[0], tr_radec[0], br_radec[0], bl_radec[0], tl_radec[0]], [tl_radec[1], tr_radec[1], br_radec[1], bl_radec[1], tl_radec[1]])
            plt.savefig(exposure.path.joinpath("stars.png"))
            print(exposure.path.joinpath("stars.png"))

        gaia_stars_skycoords = gaia_stars_skycoords[gaia_stars_inside]
        gaia_stars_df = gaia_stars_df.iloc[gaia_stars_inside]
        logger.info("Total Gaia stars in the quadrant footprint={}".format(len(gaia_stars_df)))

        # Project contained Gaia stars into pixel space
        gaia_stars_x, gaia_stars_y = gaia_stars_skycoords.to_pixel(wcs)
        gaia_stars_df['x'] = gaia_stars_x
        gaia_stars_df['y'] = gaia_stars_y

        # Removing measures that are too close to each other
        # Min distance should be a function of seeing idealy
        # aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['flag']==0]
        # aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['gflag']==0]
        min_dist = 20.
        n = len(aperse_cat.df)
        X = np.tile(aperse_cat.df['x'].to_numpy(), (n, 1))
        Y = np.tile(aperse_cat.df['y'].to_numpy(), (n, 1))
        dist = np.sqrt((X-X.T)**2+(Y-Y.T)**2)
        dist_mask = (dist <= min_dist)
        sp = dok_matrix(dist_mask)
        keys = list(filter(lambda x: x[0]!=x[1], list(sp.keys())))
        too_close_idx = list(set(list(chain(*keys))))
        keep_idx = list(filter(lambda x: x not in too_close_idx, range(n)))

        logger.info("aperse catalog: {} measures".format(n))
        logger.info("Out of which, {} are too close to each other (min distance={})".format(len(too_close_idx), min_dist))
        logger.info("{} measures are kept".format(len(keep_idx)))

        aperse_cat.df = aperse_cat.df.iloc[keep_idx]

        i = match_pixel_space(gaia_stars_df[['x', 'y']].to_records(), aperse_cat.df[['x', 'y']].to_records(), radius=0.5)
        gaia_indices = i[i>=0]
        cat_indices = np.arange(len(aperse_cat.df))[i>=0]

        standalone_stars_df = aperse_cat.df.iloc[cat_indices]
        logger.info("Old star count={}".format(len(standalone_stars_cat.df)))
        logger.info("New star count={}".format(len(standalone_stars_df)))

        old_cat = exposure.get_catalog("standalone_stars.list")
        # exposure.path.joinpath("standalone_stars.list").rename("standalone_stars.old.list")
        standalone_stars_cat.df = standalone_stars_df
        standalone_stars_cat.write()

        draw_star_shape = False
        if draw_star_shape:
            print(exposure.name)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            import numpy as np

            aperse_cat = exposure.get_catalog("aperse.list")
            standalone_stars_cat = exposure.get_catalog("standalone_stars.list")
            x, y, _, sigma_x, sigma_y, corr = aperse_cat.header['starshape']


            # plt.plot(gaia_stars_df.iloc[gaia_indices]['Gmag'].to_numpy(), (standalone_stars_cat.df['apfl6']/standalone_stars_cat.df['eapfl6']).to_numpy(), '.')
            # plt.show()

            # plt.plot(standalone_stars_cat.df['flux'].to_numpy(), standalone_stars_cat.df['fluxmax'].to_numpy(), '.')
            # plt.show()

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8., 8.))
            plt.suptitle("$N_s={}$, seeing={}\n{}".format(len(standalone_stars_cat.df), standalone_stars_cat.header['seeing'], exposure.name))
            ax.add_patch(Ellipse((x, y), width=5.*sigma_x, height=5.*sigma_y, fill=False, color='red'))
            plt.plot(np.sqrt(aperse_cat.df['gmxx'].to_numpy()), np.sqrt(aperse_cat.df['gmyy'].to_numpy()), '.', label="SE cat")
            plt.plot(np.sqrt(standalone_stars_cat.df['gmxx'].to_numpy()), np.sqrt(standalone_stars_cat.df['gmyy'].to_numpy()), '.', color='red', label="Stand. cat")
            plt.plot(np.sqrt(old_cat.df['gmxx'].to_numpy()), np.sqrt(old_cat.df['gmyy'].to_numpy()), 'x', color='green', label="Old stand. cat")
            plt.xlabel("$\\sqrt{M_g^{xx}}$")
            plt.ylabel("$\\sqrt{M_g^{yy}}$")
            plt.legend()
            plt.plot([x], [y], 'x')
            plt.xlim(0., 4.)
            plt.ylim(0., 4.)
            plt.grid()
            plt.savefig(exposure.path.joinpath("smp.png"))
            plt.close()

    return exposure.path.joinpath("standalone_stars.list").exists()


mkcat2_rm = []


def makepsf(exposure, logger, args):
    from utils import get_header_from_quadrant_path

    run_and_log(["makepsf", exposure.path, "-f"], logger)

    logger.info("Dumping header content")
    exposure.update_exposure_header()

    return exposure.path.joinpath("psfstars.list").exists()

makepsf_rm = ["psf_resid_tuple.fit", "psf_res_stack.fits", "psf_resid_image.fits", "psf_resid_tuple.dat"]


def preprocess(exposure, logger, args):
    def _run_step(f, step_name):
        logger.info("Running {}".format(step_name))
        if not f(exposure, logger, args):
            exposure.path.joinpath("{}.fail".format(step_name)).touch()
            return False
        else:
            exposure.path.joinpath("{}.success".format(step_name)).touch()
            return True

    if not _run_step(make_catalog, "make_catalog"):
        return False

    if not _run_step(mkcat2, "mkcat2"):
        return False

    return _run_step(makepsf, "makepsf")

preprocess_rm = make_catalog_rm + mkcat2_rm + makepsf_rm
