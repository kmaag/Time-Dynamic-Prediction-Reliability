from defs_global    import SETUP 
from help_functions import gen_list_starts_gt
from functions      import comp_con_comp, comp_t_s_con_comp, t_s_con_comp_plots, comp_t_s_metrics,\
                           scatter_img, regression_img, regr_classif, regr_classif_img




def main():
  """ 
  It is assumed that the hdf5 files in DIR_PROBABILITIES contain following data:
    - a three-dimensional numpy array (height, width, classes) that contains the softmax probabilities computed for the current image
    - the full filepath of the current input image
    - a two-dimensional numpy array (height, width) that contains the ground truth class indices for the current image
  """
  
  """ 
  Results: connected components (pickle files) in DIR_CON_COMPONENTS.
  """
  if SETUP.COMP_CONNECTED_COMPONENTS:
    gen_list_starts_gt()
    comp_con_comp().comp_con_comp_per_image()
    
    
  """ 
  Results: time series connected components (pickle files) in DIR_COMPONENTS_T_S.
  """  
  if SETUP.COMP_TIME_SERIES_CONNECTED_COMPONENTS:
    comp_t_s_con_comp().comp_t_s_con_comp_per_image()
    
  
  """ 
  Results: images of time series connected components (png files) in DIR_IMAGES_T_S. 
  """
  if SETUP.IMG_TIME_SERIES_CONNECTED_COMPONENTS:
    t_s_con_comp_plots().t_s_con_comp_per_image_plot()
    
    
  """ 
  Results: time series metrics (pickle files) in DIR_METRICS.
  """    
  if SETUP.COMP_TIME_SERIES_METRICS:
    comp_t_s_metrics().comp_t_s_metrics_per_image()
    
  
  """ 
  Results: images of mean lifetime (png files) in DIR_IMAGES_SCATTER. 
  """ 
  if SETUP.IMG_SCATTER_PLOTS:
    scatter_img().scatter_mean_img()
    
    
  """ 
  Results: images of regression (png files) in DIR_IMAGES_REGRESSION. Refer to paper for interpretation.
  """
  if SETUP.IMG_REGRESSION:
    regression_img().regression_per_image() 

  
  """ 
  Results: images of regression and classification (png/pickle files) in DIR_ANALYZE. 
  Choose the augmented and pseudo fators like this:
  R    0      0
  RA   m-1    0 
  RAP  m/2-1  m/2
  RP   0      m-1
  P    0      m
  """
  if SETUP.ANALYZE_METRICS:
    regr_classif().regr_classif_t_s_metrics()
    
    
  """ COMMENT:
  Results: images of analyze metrics (pdf files) in DIR_IMAGE_ANALYZE. It is very important to set PSEUDO_FACTOR to the selected value, because this is what the code is looking for.
  """
  if SETUP.IMG_ANALYZE_METRICS:
    regr_classif_img().regr_classif_img_t_s_metrics_img()
    
    

if __name__ == '__main__':
  print( "START" )
  main()
  print( "END" )
