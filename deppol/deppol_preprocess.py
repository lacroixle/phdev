#!/usr/bin/env python3

from deppol_utils import run_and_log

def make_catalog(quadrant_folder, logger, args):
    from ztfquery.io import get_file
    from shutil import copyfile

    logger.info("Retrieving calibrated.fits...")
    sciimg_path = get_file(quadrant_folder.name + "_sciimg.fits", downloadit=False)

    logger.info("Located at {}".format(sciimg_path))

    copyfile(sciimg_path, quadrant_folder.joinpath("calibrated.fits"))

    run_and_log(["make_catalog", quadrant_folder, "-O", "-S"], logger)

    return quadrant_folder.joinpath("se.list").exists()


def mkcat2(quadrant_path, logger, args):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import numpy as np

    from utils import read_list

    run_and_log(["mkcat2", quadrant_path, "-o"], logger)
    if quadrant_path.joinpath("standalone_stars.list").exists():
        se_list = read_list(quadrant_path.joinpath("aperse.list"))
        plt.plot(np.sqrt(se_list[1]['gmxx']), np.sqrt(se_list[1]['gmyy']), '.')
        plt.xlabel("$\sqrt{M_g^{xx}}$")
        plt.ylabel("$\sqrt{M_g^{yy}}$")
        plt.grid()
        plt.xlim(0., 10.)
        plt.ylim(0., 10.)
        plt.axis('equal')
        plt.savefig(quadrant_path.joinpath("moments_plan.png"), dpi=300.)
        plt.close()
        return True
    else:
        return False



def makepsf(quadrant_path, logger, args):
    run_and_log(["makepsf", quadrant_path, "-f"], logger)
    return quadrant_path.joinpath("psfstars.list").exists()


def pipeline(quadrant_path, logger, args):
    if not make_catalog(quadrant_path, logger, args):
        return False

    if not mkcat2(quadrant_path, logger, args):
        return False

    if not makepsf(quadrant_path, logger, args):
        return False

    return True

