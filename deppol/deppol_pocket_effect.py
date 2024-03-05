"""
Created on Feb 27, 2024
"""
import shutil
import os.path

from lightcurve import Exposure
from astropy.io import fits

from ztfsensors.data.correction_access import get_coef_model
from ztfsensors.pocket import correct_2d
from ztfsensors import PocketModel


def correction_no_io(image_ori, idx_ccd, idx_quad, mjd):
    '''
    
    :param image_ori:
    :param idx_ccd:
    :param idx_quad:
    :param mjd:
    '''
    alpha, cmax, beta, nmax = get_coef_model(idx_ccd, idx_quad, mjd)
    pocket_model = PocketModel(alpha, cmax, beta, nmax)
    cor_im, delta, mask = correct_2d(pocket_model, image_ori)
    return cor_im, delta, mask


def correction_with_io(f_ima, idx_ccd, idx_quad, mjd, logger, debug=False):
    # try open and correct image
    logger.info(f"process image: {f_ima}")
    f_open = False
    f_cor = False
    try:
        hdul = fits.open(f_ima, mode="update")
        f_open = True
        cor_im, _, _ = correction_no_io(hdul[0].data, idx_ccd, idx_quad, mjd)
        f_cor = True
    except:
        logger.exception("ERROR: pocket effect correction")        
    if f_cor:
        if debug:
            f_ori = f_ima.replace('.fits','_ori.fits')
            shutil.copy(f_ima, f_ori)
        hdul[0].data = cor_im
        # hdul[0].header["HISTORY"]=f"add correction pocket effet, version {ztfsensors.__version__}"
        hdul[0].header["HISTORY"] = f"add correction pocket effet"
        hdul.flush()
    if f_open:
        hdul.close()


def pocket_effect_cor(exposure, logger, args):
    """deppol interface

    :param exposure:
    :param logger:
    :param args:
    """
    try:
        image_path = exposure.retrieve_exposure(ztfin2p3_path=args.ztfin2p3_path)
    except FileNotFoundError as e:
        print(e)
        logger.error(e)
        return False
    logger.info("Found at {}".format(image_path))
    assert isinstance(exposure, Exposure)
    # extract image information
    idx_ccd = exposure.ccdid()
    idx_quad = exposure.qid()
    mjd = exposure.mjd()
    f_ima = os.path.join(image_path, "calibrated.fits")
    correction_with_io(f_ima, idx_ccd, idx_quad, mjd, debug=True)


if __name__ == '__main__':
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger= logging.getLogger()
    correction_with_io("/sps/ztf/data/storage/scenemodeling/pol_fields/test_pe/ztf_20180627482014_000600_zg_c05_o_q1/calibrated.fits",
                       5,1,0,logger,True)
