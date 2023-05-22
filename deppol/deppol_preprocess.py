#!/usr/bin/env python3

from deppol_utils import run_and_log


def make_catalog(exposure, logger, args):
    from ztfquery.io import get_file
    from shutil import copyfile
    from utils import get_header_from_quadrant_path
    import pathlib

    logger.info("Retrieving science exposure...")
    try:
        image_path = exposure.retrieve_exposure()
    except FileNotFoundError as e:
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

    run_and_log(["mkcat2", exposure.path, "-o"], logger)

    if args.use_gaia_stars:
        # Find standalone stars using Gaia
        logger.info("Using Gaia catalog to identify stars")
        aperse_cat = exposure.get_catalog("aperse.list")
        standalone_stars_cat = exposure.get_catalog("standalone_stars.list")
        gaia_stars_df = exposure.get_ext_catalog('gaia', project=True)[['x', 'y', 'Gmag']].dropna()

        # Removing measures that are too close to each other
        # Min distance should be a function of seeing idealy
        aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['flag']==0]
        aperse_cat.df = aperse_cat.df.loc[aperse_cat.df['gflag']==0]
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

        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 1)
        # plt.hist(standalone_stars_cat.df['neid'], bins='auto')
        # plt.subplot(1, 2, 2)
        # plt.hist(standalone_stars_df['neid'], bins='auto')
        # plt.show()

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
    if not make_catalog(exposure, logger, args):
        return False

    if not mkcat2(exposure, logger, args):
        return False

    return makepsf(exposure, logger, args)

preprocess_rm = make_catalog_rm + mkcat2_rm + makepsf_rm
