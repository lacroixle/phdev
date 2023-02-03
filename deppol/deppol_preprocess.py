#!/usr/bin/env python3

from deppol_utils import run_and_log


def load_calibrated(quadrant_path, ztfname, filtercode, logger, args):
    from ztfquery.io import get_file
    from shutil import copyfile
    from utils import get_header_from_quadrant_path
    import pathlib

    logger.info("Retrieving science image...")
    if not args.use_raw:
        image_path = pathlib.Path(get_file(quadrant_path.name + "_sciimg.fits", downloadit=False))

    logger.info("Located at {}".format(image_path))
    if not image_path.exists():
        logger.error("Science image does not exist on disk!")
        return False

    copyfile(image_path, quadrant_path.joinpath("calibrated.fits"))

    logger.info("Dumping header content")
    hdr = get_header_from_quadrant_path(quadrant_path)
    with open(quadrant_path.joinpath("calibrated_hdr"), 'wb') as f:
        hdr.tofile(f, sep='\n', overwrite=True, padding=False)

    return True


def make_catalog(quadrant_path, ztfname, filtercode, logger, args):
    from ztfquery.io import get_file
    from shutil import copyfile
    from utils import get_header_from_quadrant_path
    import pathlib

    logger.info("Retrieving science image...")
    if not args.use_raw:
        image_path = pathlib.Path(get_file(quadrant_path.name + "_sciimg.fits", downloadit=False))

    logger.info("Located at {}".format(image_path))
    if not image_path.exists():
        logger.error("Science image does not exist on disk!")
        return False

    copyfile(image_path, quadrant_path.joinpath("calibrated.fits"))

    run_and_log(["make_catalog", quadrant_path, "-O", "-S"], logger)

    logger.info("Dumping header content")
    hdr = get_header_from_quadrant_path(quadrant_path)
    with open(quadrant_path.joinpath("calibrated_hdr"), 'wb') as f:
        hdr.tofile(f, sep='\n', overwrite=True, padding=False)

    return quadrant_path.joinpath("se.list").exists()


make_catalog_rm = ["low.fits.gz", "miniback.fits", "segmentation.cv.fits", "segmentation.fits"]


def mkcat2(quadrant_path, ztfname, filtercode, logger, args):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import numpy as np

    from utils import read_list

    run_and_log(["mkcat2", quadrant_path, "-o"], logger)
    # if quadrant_path.joinpath("standalone_stars.list").exists():
    #     se_list = read_list(quadrant_path.joinpath("aperse.list"))
    #     plt.plot(np.sqrt(se_list[1]['gmxx']), np.sqrt(se_list[1]['gmyy']), '.')
    #     plt.xlabel("$\sqrt{M_g^{xx}}$")
    #     plt.ylabel("$\sqrt{M_g^{yy}}$")
    #     plt.grid()
    #     plt.xlim(0., 10.)
    #     plt.ylim(0., 10.)
    #     plt.axis('equal')
    #     plt.savefig(quadrant_path.joinpath("moments_plan.png"), dpi=300.)
    #     plt.close()
    #     return True
    # else:
    #     return False

    return quadrant_path.joinpath("standalone_stars.list").exists()


mkcat2_rm = []


def makepsf(quadrant_path, ztfname, filtercode, logger, args):
    from utils import get_header_from_quadrant_path

    run_and_log(["makepsf", quadrant_path, "-f"], logger)

    logger.info("Dumping header content")
    hdr = get_header_from_quadrant_path(quadrant_path)
    with open(quadrant_path.joinpath("calibrated_hdr"), 'wb') as f:
        hdr.tofile(f, sep='\n', overwrite=True, padding=False)

    return quadrant_path.joinpath("psfstars.list").exists()


#makepsf_rm = ["psf_resid_tuple.fit", "psf_res_stack.fits", "psf_resid_image.fits", "psf_resid_tuple.dat", "calibrated.fits", "weight.fz"]
makepsf_rm = ["psf_resid_tuple.fit", "psf_res_stack.fits", "psf_resid_image.fits", "psf_resid_tuple.dat"]


def preprocess(quadrant_path, ztfname, filtercode, logger, args):
    if not make_catalog(quadrant_path, logger, args):
        return False

    if not mkcat2(quadrant_path, logger, args):
        return False

    return makepsf(quadrant_path, logger, args)


pipeline_rm = make_catalog_rm + mkcat2_rm + makepsf_rm
