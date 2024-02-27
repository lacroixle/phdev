'''
Created on Feb 27, 2024

'''

def pocket_effect_cor(exposure, logger, args):
   
    from ztfsensors import pocket, PocketModel
    
    idx_ccd ,idx_quad = 1,1
    alpha,cmax,beta,nmax =get_coef_model(idx_ccd ,idx_quad)
  
    pocket_model = PocketModel(alpha,cmax,beta,nmax)
    
    image_with_pe = get_image(idx_ccd ,idx_quad)
    
    cs, delta, mask = pocket.correct_2d(pocket_model, image_with_pe)
