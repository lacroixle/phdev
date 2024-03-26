"""
Created on Feb 27, 2024
"""

import shutil
import os.path

from astropy.io import fits

from lightcurve import Exposure
from ztfsensors.data.correction_access import get_coef_model
from ztfsensors.pocket import correct_2d
from ztfsensors import PocketModel


def correction_no_io(image_ori, idx_ccd, idx_quad, mjd):
    """

    :param image_ori:
    :param idx_ccd:
    :param idx_quad:
    :param mjd:
    """
    alpha, cmax, beta, nmax = get_coef_model(idx_ccd, idx_quad, mjd)
    pocket_model = PocketModel(alpha, cmax, beta, nmax)
    cor_im, delta, mask = correct_2d(pocket_model, image_ori)
    return cor_im, delta, mask


def correction_with_io(pf_ima, idx_ccd, idx_quad, mjd, logger, debug=False):
    '''
    
    :param pf_ima: path/file sky image
    :param idx_ccd:
    :param idx_quad:
    :param mjd:
    :param logger:
    :param debug:
    '''
    logger.info(f"process image: {pf_ima}")
    b_open = False
    b_cor = False
    try:
        # try open and correct image
        hdul = fits.open(pf_ima, mode="update")
        b_open = True
        cor_im, _, _ = correction_no_io(hdul[0].data, idx_ccd, idx_quad, mjd)
        b_cor = True
    except:
        logger.exception("ERROR: pocket effect correction")
    if b_cor:
        if debug:
            pf_ori = pf_ima.replace(".fits", "_ori.fits")
            # prevent remove original image
            if not os.path.exists(pf_ori):
                shutil.copy(pf_ima, pf_ori)
        # update image, history and write it with flush()
        hdul[0].data = cor_im
        hdul[0].header["HISTORY"] = f"add correction pocket effet"
        hdul.flush()
    if b_open:
        hdul.close()


def pocket_effect_cor(exposure, logger, args):
    """deppol interface

    :param exposure:
    :param logger:
    :param args:
    """
    assert isinstance(exposure, Exposure)
    # extract image information
    pf_ima = os.path.join(exposure.path, "calibrated.fits")
    idx_ccd = exposure.ccdid()
    idx_quad = exposure.qid()
    mjd = exposure.mjd()
    # process correction
    correction_with_io(pf_ima, idx_ccd, idx_quad, mjd, debug=True)


def pocket_effect_metric(exposure, logger, args):
    """deppol interface

    :param exposure:
    :param logger:
    :param args:
    """
    assert isinstance(exposure, Exposure)
    f_cat_stst = exposure.path.joinpath("standalone_stars.list")
    cat_stst = exposure.get_catalog(f_cat_stst)


def pocket_effect_test(exposure, logger, args):
    """deppol interface

    :param exposure:
    :param logger:
    :param args:
    """
    assert isinstance(exposure, Exposure)
    # logger.info("Found at {}".format(image_path))
    # image_path.joinpath('pe_test').touch()
    # print(image_path)
    print(exposure.path, exposure.name)


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger()
    correction_with_io(
        "/sps/ztf/data/storage/scenemodeling/pol_fields/test_pe/ztf_20180627482014_000600_zg_c05_o_q1/calibrated.fits",
        5,
        1,
        0,
        logger,
        True,
    )
