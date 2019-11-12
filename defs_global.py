class SETUP:
  
  # set path
  
  IO_PATH   = "/home/user/seg_unc_io/"   
  
  # select
  
  CLASS_DTYPES         = [ "one_hot_classes", "probs" ] 
  CLASS_DTYPE          = CLASS_DTYPES[1]  
  IMAGE_TYPES          = [ "kitti_video", "viper_video"] 
  IMAGE_TYPE           = IMAGE_TYPES[0]  
  NET_TYPES            = [ "xc.mscl.os8", "mn.sscl.os16", "xc.sscl.os16" ]     
  NET_TYPE             = NET_TYPES[1]    
  CLASSIFICATION_TYPES = ["LR_L1", "GB", "NN_L2"]
  CLASSIFICATION_TYPE  = CLASSIFICATION_TYPES[0]  
  REGRESSION_TYPES     = ["LR", "LR_L1", "LR_L2", "GB", "NN_L1", "NN_L2"]
  REGRESSION_TYPE      = REGRESSION_TYPES[0]        
  TRAIN_TYPES          = ["R", "RA", "RAP", "RP", "P"]   
  TRAIN_TYPE           = TRAIN_TYPES[0]
  
  # select tasks to be executed by setting boolean variable True/False
  
  COMP_CONNECTED_COMPONENTS             = False 
  COMP_TIME_SERIES_CONNECTED_COMPONENTS = False
  IMG_TIME_SERIES_CONNECTED_COMPONENTS  = False
  COMP_TIME_SERIES_METRICS              = False
  IMG_SCATTER_PLOTS                     = False
  IMG_REGRESSION                        = False 
  ANALYZE_METRICS                       = False
  IMG_ANALYZE_METRICS                   = False
  
  # optionals
  
  NUM_CORES            = 1
  TRACKING_EPSILON     = 100
  TRACKING_NUMBER_REGR = 5
  NUMBER_OF_FRAMES     = 10
  AUGMENTED_FACTOR     = 0   
  PSEUDO_FACTOR        = 0
  TRAIN_SPLIT          = 70  
  VAL_SPLIT            = 10 
  TEST_SPLIT           = 20 
  NUMBER_OF_RUNS       = 10
  CLASSIFICATION_FLAG  = 0
  
  if IMAGE_TYPE == IMAGE_TYPES[0]:
    NUM_IMAGES = 12223
  elif IMAGE_TYPE == IMAGE_TYPES[1]:
    TRAIN_TYPE = "R"
    NUM_IMAGES = 3593
  
  TMP_PATH                      = "_nr" + str(NUMBER_OF_FRAMES) + "_train" + str(TRAIN_TYPE) + "_fa" + str(AUGMENTED_FACTOR) + "_fp" + str(PSEUDO_FACTOR) + "/"
  DIR_IMAGES                    = IO_PATH + IMAGE_TYPE + "/inputimages/val/"
  DIR_GROUND_TRUTH              = IO_PATH + IMAGE_TYPE + "/groundtruth/val/"
  DIR_PROBABILITIES             = IO_PATH + IMAGE_TYPE + "/probs/"                         + NET_TYPE + "/"
  DIR_CON_COMPONENTS            = IO_PATH + IMAGE_TYPE + "/components/"                    + NET_TYPE + "/"
  DIR_COMPONENTS_T_S            = IO_PATH + IMAGE_TYPE + "/components_time_series/"        + NET_TYPE + "/"
  DIR_IMAGES_T_S                = IO_PATH + IMAGE_TYPE + "/components_time_series_images/" + NET_TYPE + "/"
  DIR_METRICS                   = IO_PATH + IMAGE_TYPE + "/metrics/"                       + NET_TYPE + "/"
  DIR_IMAGES_SCATTER            = IO_PATH + IMAGE_TYPE + "/scatter_images/"                + NET_TYPE + "/"
  DIR_IMAGES_REGRESSION         = IO_PATH + IMAGE_TYPE + "/regresssion_images"  + TMP_PATH + NET_TYPE + "/"
  DIR_ANALYZE                   = IO_PATH + IMAGE_TYPE + "/analyze"             + TMP_PATH + NET_TYPE + "_runs" + str(NUMBER_OF_RUNS) + "/"
  DIR_IMAGE_ANALYZE             = IO_PATH + IMAGE_TYPE + "/analyze_images/"                + NET_TYPE + "_runs" + str(NUMBER_OF_RUNS) + "/"
  DIR_AUGMENTED                 = IO_PATH + IMAGE_TYPE + "/metrics/upsampling"  + TMP_PATH + NET_TYPE + "/" + REGRESSION_TYPE + "_runs" + str(NUMBER_OF_RUNS) + "/" 
  

  
"""
In case of problems, feel free to contact: Kira Maag, maag@math.uni-wuppertal.de
"""
