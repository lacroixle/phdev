"""
Created on Feb 27, 2024

"""
import shutil
import os.path

from lightcurve import Exposure
from astropy.io import fits

from ztfsensors import pocket, PocketModel


def correction_no_io(image_ori, idx_ccd, idx_quad, mjd):
    alpha, cmax, beta, nmax = get_coef_model(idx_ccd, idx_quad, mjd)
    pocket_model = PocketModel(alpha, cmax, beta, nmax)
    cor_im, delta, mask = pocket.correct_2d(pocket_model, image_ori)
    return cor_im, delta, mask


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
    f_ima = os.path.join(image_path,"calibrated.fits")
    # try open and correct image
    f_open = False
    f_cor = False
    try:
        hdul = fits.open(f_ima)
        f_open = True
        cor_im, _, _ = correction_no_io(hdul[0].data, idx_ccd, idx_quad, mjd)
        f_cor = True
    except:        
        logger.error("ERROR: pocket effect correction")

    if f_cor:
        shutil.copy(f_ima,os.path.join(image_path,"calibrated_nocor.fits") )
        hdul[0].data = cor_im
        hdul.flush()
    
    if f_open:
        hdul.close()
        
        
        
    
