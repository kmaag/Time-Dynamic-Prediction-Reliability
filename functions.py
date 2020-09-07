import time
import os
import pickle
import numpy as np
import matplotlib.colors as colors
from multiprocessing import Pool
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc

from defs_global          import SETUP
from regr_classif         import classification, regression
from components_metrics   import compute_con_comp, comp_t_s_con_comp_per_video, compute_t_s_metrics
from help_functions       import name_to_latex, t_s_metrics_concat, t_s_metrics_dataset,\
                                 train_val_test_split, t_s_dataset_concat     
from plotting_function    import plot_matching, scatter_lifetime_plot, t_s_metrics_i_img,\
                                 plot_regression_scatter, timeline_img, time_r2_auc_img
from in_and_out_functions import get_save_path_probs_i, probs_gt_load, con_comp_dump, t_s_con_comp_load,\
                                 get_save_path_t_s_con_comp_i, get_save_path_t_s_metrics_i,\
                                 t_s_metrics_dump, t_s_metrics_load

                                 
                          
class comp_con_comp(object):

  def __init__(self, number_cores=1, rewrite=True):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.rewrite = rewrite
    self.number_images = SETUP.NUM_IMAGES
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    :param rewrite: (boolean) overwrite existing files if True
    """

  def comp_con_comp_per_image(self):
    print("calculating connected components")
    p_args = [ (k,) for k in range(self.number_images) ]
    Pool(self.number_cores).starmap( self.comp_con_comp_i, p_args )
    
  def comp_con_comp_i(self,i):
    if os.path.isfile( get_save_path_probs_i(i) ) and self.rewrite:
      start = time.time()
      probs, gt, _ = probs_gt_load( i )
      con_comp = compute_con_comp( probs, gt )
      con_comp_dump( con_comp, i )
      print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )  
      
      
      
class comp_t_s_con_comp(object):

  def __init__(self, number_cores=1):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.number_images = SETUP.NUM_IMAGES
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    """

  def comp_t_s_con_comp_per_image(self):
    print("calculating time series con_comp")
    start_path = SETUP.DIR_PROBABILITIES + "../list_start.npy"
    start_list = np.load(start_path)
    stop_list = []
    for i in start_list[1:len(start_list)]:
      stop_list.append( int(i)-1 )
    stop_list.append( self.number_images-1 )

    p_args = [ (start_list[k],stop_list[k]) for k in range(0,len(start_list)) ]
    Pool(self.number_cores).starmap( comp_t_s_con_comp_per_video, p_args ) 
    
    

class t_s_con_comp_plots(object):
  
  def __init__(self, number_cores=1):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.number_images = SETUP.NUM_IMAGES
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    """

  def t_s_con_comp_per_image_plot(self):
    print("plot time series con_comp")
    save_path = SETUP.DIR_IMAGES_T_S
    if not os.path.exists( save_path ):
      os.makedirs( save_path )
    colors_list_tmp = list(colors._colors_full_map.values())  # 1163 colors
    colors_list = []
    for color in colors_list_tmp:
      if len(color) == 7:
        colors_list.append(color)
    start_path = SETUP.DIR_PROBABILITIES + "../list_start.npy"
    start_list = np.load(start_path)
    image_list = []
    for i in range(1,self.number_images):
      if (i not in start_list) and ((i-1) not in start_list):
        image_list.append( i-1 )
    p_args = [ (k,colors_list) for k in image_list ]
    Pool(self.number_cores).starmap( plot_matching, p_args ) 



class comp_t_s_metrics(object):

  def __init__(self, number_cores=1):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.number_images = SETUP.NUM_IMAGES
    self.epsilon = SETUP.TRACKING_EPSILON
    self.num_reg = SETUP.TRACKING_NUMBER_REGR
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    :param epsilon: (int) used in matching algorithm
    :param num_reg: (int) used in matching algorithm
    """

  def comp_t_s_metrics_per_image(self):
    print("calculating time series metrics")
    max_con_comp = 0
    for i in range(self.number_images):
      t_s_con_comp = t_s_con_comp_load( i, self.epsilon, self.num_reg )
      tmp_number = -int(t_s_con_comp.min())
      if tmp_number > max_con_comp:
        max_con_comp = tmp_number
    p_args = [ (k,max_con_comp) for k in range(self.number_images) ]
    Pool(self.number_cores).starmap( self.comp_t_s_metrics_i, p_args ) 
    
  def comp_t_s_metrics_i( self, i, max_con_comp ):
    if os.path.isfile( get_save_path_t_s_con_comp_i( i, self.epsilon, self.num_reg ) ) and not os.path.isfile( get_save_path_t_s_metrics_i( i, self.epsilon, self.num_reg ) ):
      start = time.time()
      t_s_con_comp = t_s_con_comp_load( i, self.epsilon, self.num_reg )
      probs, gt, _ = probs_gt_load( i ) 
      t_s_metrics = compute_t_s_metrics( t_s_con_comp, probs, gt, max_con_comp )
      t_s_metrics_dump( t_s_metrics, i, self.epsilon, self.num_reg ) 
      print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )
      


class scatter_img(object):

  def __init__(self):
    self.number_images = SETUP.NUM_IMAGES
    self.epsilon = SETUP.TRACKING_EPSILON
    self.num_reg = SETUP.TRACKING_NUMBER_REGR
    """
    object initialization
    :param number_images: (int) number of images to be processed
    :param epsilon: (int) used in matching algorithm
    :param num_reg: (int) used in matching algorithm
    """

  def scatter_mean_img(self):
    print("visualize lifetime of connected con_comp")
    
    size_cut = 1000
    start_path = SETUP.DIR_PROBABILITIES + "../list_start.npy"
    start_list = np.load(start_path)
    starts = np.zeros((len(start_list)))
    starts[0:len(start_list)-1] = start_list[1:len(start_list)]
    starts[len(start_list)-1] = self.number_images
    max_con_comp = 0
    end_list = []
    
    for i in starts:
      end_list.append( int(i)-1 )
      t_s_con_comp = t_s_con_comp_load( int(i-1), self.epsilon, self.num_reg )
      tmp_number = -int(t_s_con_comp.min())
      if tmp_number > max_con_comp:
        max_con_comp = tmp_number
    lifetime_mean_size = np.zeros(( max_con_comp*len(start_list), 2 ))
    
    for k in range(len(start_list)):
      for j in range(start_list[k], end_list[k]+1):
        t_s_metrics = t_s_metrics_load ( j, self.epsilon, self.num_reg )
        for i in range(0,max_con_comp):
          if t_s_metrics["S_in"][i] > 0:
            lifetime_mean_size[k*max_con_comp+i,0] += 1
            lifetime_mean_size[k*max_con_comp+i,1] += t_s_metrics["S_in"][i]
            
    for i in range(0,max_con_comp*len(start_list)):  
      if lifetime_mean_size[i,0] > 0:
        lifetime_mean_size[i,1] = lifetime_mean_size[i,1] / lifetime_mean_size[i,0]     
    idx1 = np.asarray(np.where(lifetime_mean_size[:,1] > 0))
    lifetime_mean_size_del = lifetime_mean_size[idx1[0,:],:]
    idx2 = np.asarray(np.where(lifetime_mean_size[:,1] > size_cut))
    lifetime_mean_size_del_cut = lifetime_mean_size[idx2[0,:],:]
    mean_lifetime = np.zeros((2))
    mean_lifetime[0] = np.sum(lifetime_mean_size_del[:,0]) / lifetime_mean_size_del.shape[0]
    mean_lifetime[1] = np.sum(lifetime_mean_size_del_cut[:,0]) / lifetime_mean_size_del_cut.shape[0]
    print("mean lifetime:", mean_lifetime[0], ", mean lifetime for con_comp > ", size_cut, ":", mean_lifetime[1])
    scatter_lifetime_plot( mean_lifetime, lifetime_mean_size_del, size_cut, lifetime_mean_size_del_cut )



class regression_img(object):
  
  def __init__(self, number_cores=1):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.number_images = SETUP.NUM_IMAGES
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    """

  def regression_per_image(self):
    print("visualization running")
    t= time.time()
    np.random.seed( 0 )
    
    metrics, start = t_s_metrics_concat( )
    nclasses = np.max( metrics["class"] ) + 1
    Xa, classes, ya, _, _, _ = t_s_metrics_dataset( metrics, nclasses )
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    Xa_train, ya_train, Xa_val, ya_val, ya_zero_val, max_num_prev_frames, not_del_rows_val, plot_image_list = self.prepare_regression_data( Xa, ya, start ) 
    y_train_pred, y_val_pred, _, _, _ = regression(Xa_train, ya_train, Xa_val, ya_val, Xa_val, Xa_val, max_num_prev_frames) 
    print("time series model r2 score (train):", r2_score(ya_train,y_train_pred) )
    print("time series model r2 score (val):", r2_score(ya_val,y_val_pred) )
    print(" ")
      
    ya_pred = np.zeros((ya_zero_val.shape[0]))
    counter = 0
    for i in range(ya_zero_val.shape[0]):
      if not_del_rows_val[i] == True:
        ya_pred[i] = y_val_pred[counter]
        counter += 1
    print("Start visualize time series metrics")
    name = SETUP.REGRESSION_TYPE + "_img" 
    p_args = [ (ya_zero_val[start[1]*j:start[1]*(j+1)], ya_pred[start[1]*j:start[1]*(j+1)], i, name) for i,j in zip( list(zip(*plot_image_list))[0],  list(zip(*plot_image_list))[1] ) ]
    Pool(self.number_cores).starmap( t_s_metrics_i_img, p_args )
    print("time needed ", time.time()-t)  
    
  def prepare_regression_data( self, Xa, ya, start ):
    
    epsilon = SETUP.TRACKING_EPSILON
    num_reg = SETUP.TRACKING_NUMBER_REGR
    
    ya = np.squeeze(ya)
    Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, max_num_prev_frames = t_s_dataset_concat( Xa, ya, ya, start  )
    Xa_train, _, _, _, ya_train, _, _, _, _, _, _, _ = train_val_test_split( Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P )
    
    start_path = SETUP.DIR_PROBABILITIES + "../list_start.npy"
    start_list = np.load(start_path)
    starts = np.zeros((len(start_list)+1))
    starts[0:len(start_list)] = start_list[0:len(start_list)]
    starts[len(start_list)] = self.number_images
    
    plot_image_list = []
    Xa_zero_val = np.zeros(( (self.number_images-(len(start_list)*max_num_prev_frames)) * start[1], Xa.shape[1] * (max_num_prev_frames+1)))
    ya_zero_val = np.zeros(( (self.number_images-(len(start_list)*max_num_prev_frames)) * start[1] ))
    counter = 0
    for c,k in zip(starts[0:len(start_list)], range(len(start_list))):
      for i in range(int(c+max_num_prev_frames),int(starts[k+1])):
        plot_image_list.append( ( i, counter ) )
        tmp = np.zeros(( start[1], Xa.shape[1] * (max_num_prev_frames+1) ))
        for j in range(0,max_num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[start[i-j]:start[i-j+1]]  
        Xa_zero_val[start[1]*counter:start[1]*(counter+1),:] = tmp
        ya_zero_val[start[1]*counter:start[1]*(counter+1)] = ya[start[i]:start[i+1]]
        counter +=1

    not_del_rows_val = ~(Xa_zero_val[:,0:Xa.shape[1]]==0).all(axis=1)
    Xa_val = Xa_zero_val[not_del_rows_val]
    ya_val = ya_zero_val[not_del_rows_val]
    ya_val = np.squeeze(ya_val)
    ya_zero_val = np.squeeze(ya_zero_val)
    print("shapes:", "Xa train", np.shape(Xa_train), "ya train", np.shape(ya_train), "Xa val",  np.shape(Xa_val), "ya val", np.shape(ya_val))
    return Xa_train, ya_train, Xa_val, ya_val, ya_zero_val, max_num_prev_frames, not_del_rows_val, plot_image_list
  
  
     
class regr_classif(object):

  def __init__(self, number_cores=1):
    self.number_cores = number_cores if not hasattr(SETUP, 'NUM_CORES') else SETUP.NUM_CORES
    self.number_images = SETUP.NUM_IMAGES
    """
    object initialization
    :param number_cores: (int) number of cores used for parallelization
    :param number_images: (int) number of images to be processed
    """
    
  def regr_classif_t_s_metrics( self ):
    print("start analyzing")
    t= time.time()
    runs = SETUP.NUMBER_OF_RUNS
    
    if not os.path.exists( SETUP.DIR_ANALYZE+'scatter/' ):
      os.makedirs( SETUP.DIR_ANALYZE+'scatter/' )
      os.makedirs( SETUP.DIR_ANALYZE+'stats/' )

    metrics, start = t_s_metrics_concat( )
    nclasses = np.max( metrics["class"] ) + 1
    Xa, classes, ya, y0a, X_names, class_names = t_s_metrics_dataset( metrics, nclasses )
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    X_names += class_names
    Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, max_num_prev_frames = t_s_dataset_concat( Xa, ya, y0a, start )

    classification_stats = ['penalized_val_acc', 'penalized_val_auroc', 'penalized_train_acc', 'penalized_train_auroc', 'penalized_test_R_acc', 'penalized_test_R_auroc', 'penalized_test_P_acc', 'penalized_test_P_auroc' ]
    regression_stats        = ['regr_val_mse', 'regr_val_r2', 'regr_train_mse', 'regr_train_r2', 'regr_test_R_mse', 'regr_test_R_r2','regr_test_P_mse', 'regr_test_P_r2' ]
    
    stats = self.initialize_statistics( runs, X_names, classification_stats, regression_stats, max_num_prev_frames )
  
    print("start runs")
    if "LR" in SETUP.REGRESSION_TYPE and (SETUP.CLASSIFICATION_TYPE == "LR_L1" or SETUP.CLASSIFICATION_FLAG == 0):  
      single_run_stats = self.initialize_statistics( runs, X_names, classification_stats, regression_stats, max_num_prev_frames )
      p_args = [ ( Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, X_names, Xa.shape[1], max_num_prev_frames, single_run_stats, run ) for run in range(runs) ]
      single_run_stats = Pool(self.number_cores).starmap( self.regr_classif_runs, p_args )
      for num_frames in range(max_num_prev_frames+1):
        for run in range(runs):
          for s in stats:
            if s not in [ "metric_names"]:
              stats[s][num_frames][run] = single_run_stats[run][s][num_frames][run] 
    else:
      for run in range(runs):
        stats = self.regr_classif_runs(Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, X_names, Xa.shape[1], max_num_prev_frames, stats, run)
    
    pickle.dump( stats, open( SETUP.DIR_ANALYZE + 'stats/' + SETUP.REGRESSION_TYPE + "_stats.p", "wb" ) )
    if SETUP.CLASSIFICATION_FLAG == 1:
      pickle.dump( stats, open( SETUP.DIR_ANALYZE + 'stats/' + SETUP.CLASSIFICATION_TYPE + "_CL_stats.p", "wb" ) ) 
    print("regression (and classification) finished")
    
    mean_stats = dict({})
    std_stats = dict({})
    for s in classification_stats:
      mean_stats[s] = 0.5*np.ones((max_num_prev_frames+1))
      std_stats[s] = 0.5*np.ones((max_num_prev_frames+1))
    for s in regression_stats:
      mean_stats[s] = np.zeros((max_num_prev_frames+1,))
      std_stats[s] = np.zeros((max_num_prev_frames+1,))
    
    for num_frames in range(max_num_prev_frames+1):
      for s in stats:
        if s not in [ "metric_names"]:
          mean_stats[s][num_frames] = np.mean(stats[s][num_frames], axis=0)
          std_stats[s][num_frames]  = np.std( stats[s][num_frames], axis=0)
    
    num_timeseries = np.arange(1, max_num_prev_frames+2)
    timeline_img(num_timeseries, np.asarray(mean_stats['regr_train_r2']), np.asarray(std_stats['regr_train_r2']), np.asarray(mean_stats['regr_val_r2']), np.asarray(std_stats['regr_val_r2']), np.asarray(mean_stats['regr_test_R_r2']), np.asarray(std_stats['regr_test_R_r2']), np.asarray(mean_stats['regr_test_P_r2']), np.asarray(std_stats['regr_test_P_r2']), 'r2')
        
    if SETUP.CLASSIFICATION_FLAG == 1:
          
      timeline_img(num_timeseries, np.asarray(mean_stats['penalized_train_auroc']), np.asarray(std_stats['penalized_train_auroc']), np.asarray(mean_stats['penalized_val_auroc']), np.asarray(std_stats['penalized_val_auroc']), np.asarray(mean_stats['penalized_test_R_auroc']), np.asarray(std_stats['penalized_test_R_auroc']), np.asarray(mean_stats['penalized_test_P_auroc']), np.asarray(std_stats['penalized_test_P_auroc']), 'auc')
        
      timeline_img(num_timeseries, np.asarray(mean_stats['penalized_train_acc']), np.asarray(std_stats['penalized_train_acc']), np.asarray(mean_stats['penalized_val_acc']), np.asarray(std_stats['penalized_val_acc']), np.asarray(mean_stats['penalized_test_R_acc']), np.asarray(std_stats['penalized_test_R_acc']), np.asarray(mean_stats['penalized_test_P_acc']), np.asarray(std_stats['penalized_test_P_acc']), 'acc')
    
    print("time needed ", time.time()-t)
    
  def initialize_statistics( self, n_av, X_names, classification_stats, regression_stats, max_num_prev_frames ):
  
    n_metrics = len(X_names) * (max_num_prev_frames+1)
    stats     = dict({})
    for s in classification_stats:
      stats[s] = 0.5*np.ones((max_num_prev_frames+1,n_av))
    for s in regression_stats:
      stats[s] = np.zeros((max_num_prev_frames+1,n_av))
    stats["metric_names"] = X_names
    return stats 
  
  def regr_classif_runs( self, Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, X_names, num_metrics, max_num_prev_frames, stats, run ): 
    
    Xa_train_all, Xa_val_all, Xa_test_R_all, Xa_test_P_all, ya_train, ya_val, ya_test_R, ya_test_P, y0a_train, y0a_val, y0a_test_R, y0a_test_P = train_val_test_split( Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, run )
    
    for num_frames in range(max_num_prev_frames+1):
      
      Xa_train = Xa_train_all[:,0:(num_metrics * (num_frames+1))]
      Xa_val= Xa_val_all[:,0:(num_metrics * (num_frames+1))]
      Xa_test_R = Xa_test_R_all[:,0:(num_metrics * (num_frames+1))]
      Xa_test_P = Xa_test_P_all[:,0:(num_metrics * (num_frames+1))]
      
      print("run", run, "num frames", num_frames, "shapes:", "Xa train", np.shape(Xa_train), "Xa val", np.shape(Xa_val), "Xa test real", np.shape(Xa_test_R), "Xa test pseudo", np.shape(Xa_test_P), "ya train", np.shape(ya_train), "ya val", np.shape(ya_val), "ya test real", np.shape(ya_test_R), "ya test pseudo", np.shape(ya_test_P))

      if SETUP.CLASSIFICATION_FLAG == 1:
        
        y0a_train_pred, y0a_val_pred, y0a_test_R_pred, y0a_test_P_pred = classification( Xa_train, y0a_train, Xa_val, y0a_val, Xa_test_R, Xa_test_P )
        stats['penalized_train_acc'][num_frames,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
        stats['penalized_val_acc'][num_frames,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
        stats['penalized_test_R_acc'][num_frames,run] = np.mean( np.argmax(y0a_test_R_pred,axis=-1)==y0a_test_R )
        stats['penalized_test_P_acc'][num_frames,run] = np.mean( np.argmax(y0a_test_P_pred,axis=-1)==y0a_test_P )
        fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
        stats['penalized_train_auroc'][num_frames,run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
        stats['penalized_val_auroc'][num_frames,run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_test_R, y0a_test_R_pred[:,1])
        stats['penalized_test_R_auroc'][num_frames,run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_test_P, y0a_test_P_pred[:,1])
        stats['penalized_test_P_auroc'][num_frames,run] = auc(fpr, tpr)
            
      ya_train_pred, ya_val_pred, ya_test_R_pred, ya_test_P_pred, _ = regression( Xa_train, ya_train, Xa_val, ya_val, Xa_test_R, Xa_test_P, num_frames )
      stats['regr_train_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
      stats['regr_train_r2'][num_frames,run]  = r2_score(ya_train, ya_train_pred)
      stats['regr_val_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )
      stats['regr_val_r2'][num_frames,run]  = r2_score(ya_val, ya_val_pred)
      stats['regr_test_R_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_test_R, ya_test_R_pred) )
      stats['regr_test_R_r2'][num_frames,run]  = r2_score(ya_test_R, ya_test_R_pred)
      stats['regr_test_P_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_test_P, ya_test_P_pred) )
      stats['regr_test_P_r2'][num_frames,run]  = r2_score(ya_test_P, ya_test_P_pred)
      if run == 0:
        plot_regression_scatter( Xa_test_R, ya_test_R, ya_test_R_pred, X_names, num_frames )
        
    return stats
  
  
  
class regr_classif_img(object):

  def __init__(self):
    self.max_num_prev_frames = SETUP.NUMBER_OF_FRAMES
    self.reg_list = SETUP.REGRESSION_TYPES
    self.cl_list = SETUP.CLASSIFICATION_TYPES
    """
    object initialization
    :param max_num_prev_frames: (int) number of previous frames
    :param reg_list: (list) different regression models
    :param cl_list: (list) different classification models
    """
    
  def regr_classif_img_t_s_metrics_img( self ):
    print("visualization running")
    
    if not os.path.exists( SETUP.DIR_IMAGE_ANALYZE ):
      os.makedirs( SETUP.DIR_IMAGE_ANALYZE )
    
    if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
      n = SETUP.PSEUDO_FACTOR
      data_list = []
      data_list.append( "R" + "_fa" + str(0) + "_fp" + str(0) )
      data_list.append( "RA" + "_fa" + str(n-1) + "_fp" + str(0) )
      data_list.append( "RAP" + "_fa" + str(int(n/2-1)) + "_fp" + str(int(n/2)) )
      data_list.append( "RP" + "_fa" + str(0) + "_fp" + str(n-1) )
      data_list.append( "P" + "_fa" + str(0) + "_fp" + str(n) )
      self.min_max_file_and_time_img(data_list)       
    elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
      data = "R_fa0_fp0"
      self.min_max_file_and_time_img(data)
  
  def min_max_file_and_time_img( self, data_list ):
    data_reg_names_list = []
    data_reg_list_mean = []
    data_reg_list_std = []
    data_cl_names_list = []
    data_cl_list_mean = []
    data_cl_list_std = []
    max_r2_list = [0, 0, -1, 'empty']
    max_auc_list = [0, 0, -1, 'empty']
    max_acc_list = [0, 0, -1, 'empty']
    if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
      min_r2_list = [1, 0, 'empty']
      min_auc_list = [1, 0, 'empty']
      min_acc_list = [1, 0, 'empty']
      
    for data in data_list:
      
      read_path1 = SETUP.IO_PATH + SETUP.IMAGE_TYPE + "/analyze_nr" + str(SETUP.NUMBER_OF_FRAMES) + "_train" + data + "/" + SETUP.NET_TYPE + "_runs" + str(SETUP.NUMBER_OF_RUNS) + "/stats/"
            
      for reg_type in self.reg_list:
        
        read_path = read_path1 + reg_type + "_stats.p"
        stats = pickle.load( open( read_path, "rb" ) )
        data_reg_names_list.append(str(data)+"_"+str(reg_type))
        
        for num_frames in range(self.max_num_prev_frames+1):
          data_reg_list_mean.append( np.mean(stats["regr_test_R_r2"][num_frames], axis=0) )
          data_reg_list_std.append( np.std(stats["regr_test_R_r2"][num_frames], axis=0) )
          
          if max_r2_list[0] < np.mean(stats["regr_test_R_r2"][num_frames], axis=0):
            max_r2_list[0] = np.mean(stats["regr_test_R_r2"][num_frames], axis=0)
            max_r2_list[1] = np.std(stats["regr_test_R_r2"][num_frames], axis=0)
            max_r2_list[2] = num_frames
            max_r2_list[3] = str(data)+"_"+str(reg_type)
        
        if reg_type == "LR":
          if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
            if min_r2_list[0] > np.mean(stats["regr_test_R_r2"][0], axis=0):
                min_r2_list[0] = np.mean(stats["regr_test_R_r2"][0], axis=0)
                min_r2_list[1] = np.std(stats["regr_test_R_r2"][0], axis=0)
                min_r2_list[2] = str(data)+"_"+str(reg_type)
          elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
            min_r2_list = [np.mean(stats["regr_test_R_r2"][0], axis=0), np.std(stats["regr_test_R_r2"][0], axis=0), str(data)+"_"+str(reg_type)]
              
      for cl_type in self.cl_list:
      
        read_path = read_path1 + cl_type + "_CL_stats.p"
        stats = pickle.load( open( read_path, "rb" ) )
        data_cl_names_list.append(str(data)+"_"+str(cl_type)+"_CL")
        
        for num_frames in range(self.max_num_prev_frames+1):
          data_cl_list_mean.append( np.mean(stats["penalized_test_R_auroc"][num_frames], axis=0) )
          data_cl_list_std.append( np.std(stats["penalized_test_R_auroc"][num_frames], axis=0) )
          
          if max_auc_list[0] < np.mean(stats["penalized_test_R_auroc"][num_frames], axis=0):
            max_auc_list[0] = np.mean(stats["penalized_test_R_auroc"][num_frames], axis=0)
            max_auc_list[1] = np.std(stats["penalized_test_R_auroc"][num_frames], axis=0)
            max_auc_list[2] = num_frames
            max_auc_list[3] = str(data)+"_"+str(cl_type)+"_CL"
          if max_acc_list[0] < np.mean(stats["penalized_test_R_acc"][num_frames], axis=0):
            max_acc_list[0] = np.mean(stats["penalized_test_R_acc"][num_frames], axis=0)
            max_acc_list[1] = np.std(stats["penalized_test_R_acc"][num_frames], axis=0)
            max_acc_list[2] = num_frames
            max_acc_list[3] = str(data)+"_"+str(cl_type)+"_CL"
            
        if cl_type == "LR_L1":
          if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
            if min_auc_list[0] > np.mean(stats["penalized_test_R_auroc"][0], axis=0):
              min_auc_list[0] = np.mean(stats["penalized_test_R_auroc"][0], axis=0)
              min_auc_list[1] = np.std(stats["penalized_test_R_auroc"][0], axis=0)
              min_auc_list[2] = str(data)+"_"+str(cl_type)+"_CL"
            if min_acc_list[0] > np.mean(stats["penalized_test_R_acc"][0], axis=0):
              min_acc_list[0] = np.mean(stats["penalized_test_R_acc"][0], axis=0)
              min_acc_list[1] = np.std(stats["penalized_test_R_acc"][0], axis=0)
              min_acc_list[2] = str(data)+"_"+str(cl_type)+"_CL"
          elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
            min_auc_list = [np.mean(stats["penalized_test_R_auroc"][0], axis=0), np.std(stats["penalized_test_R_auroc"][0], axis=0), str(data)+"_"+str(cl_type)+"_CL"]
            min_acc_list = [np.mean(stats["penalized_test_R_acc"][0], axis=0), np.std(stats["penalized_test_R_acc"][0], axis=0), str(data)+"_"+str(cl_type)+"_CL"]
            
    result_path = os.path.join(SETUP.DIR_IMAGE_ANALYZE, "max_min_results.txt")
    with open(result_path, 'a') as fi:
      print("max R^2:", max_r2_list[0], "std:", max_r2_list[1], "num frames:", max_r2_list[2], "type:", max_r2_list[3], file=fi)
      print("max auroc:", max_auc_list[0], "std:", max_auc_list[1], "num frames:", max_auc_list[2], "type:", max_auc_list[3], file=fi)
      print("max accuracy:", max_acc_list[0], "std:", max_acc_list[1], "num frames:", max_acc_list[2], "type:", max_acc_list[3], file=fi)
      print("minimum with LR in regression/ LR_L1 in classification with 0 additional frames", file=fi)
      print("min R^2:", min_r2_list[0], "std:", min_r2_list[1], "num frames: 0", "type:", min_r2_list[2], file=fi)
      print("min auroc:", min_auc_list[0], "std:", min_auc_list[1], "num frames 0:", "type:", min_auc_list[2], file=fi)
      print("min accuracy:", min_acc_list[0], "std:", min_acc_list[1], "num frames 0:", "type:", min_acc_list[2], file=fi)
       
    if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
      
      for reg_type in self.reg_list:
        
        mean_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(data_list) ))
        std_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(data_list) ))
        for data, i in zip(data_list, range(len(data_list))):
          
          for ind in range(len(data_reg_names_list)):
            if (str(data)+"_"+str(reg_type)) == data_reg_names_list[ind]:
              break
          
          mean_list_data[i*(self.max_num_prev_frames+1):(i*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_reg_list_mean[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
          std_list_data[i*(self.max_num_prev_frames+1):(i*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_reg_list_std[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
              
        time_r2_auc_img(self.max_num_prev_frames, data_list, reg_type, mean_list_data, std_list_data, len(data_list), 'r2')

      for cl_type in self.cl_list:
        
        mean_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(data_list) ))
        std_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(data_list) ))
        for data, i in zip(data_list, range(len(data_list))):
          
          for ind in range(len(data_cl_names_list)):
            if (str(data)+"_"+str(cl_type)+"_CL") == data_cl_names_list[ind]:
              break
          
          mean_list_data[i*(self.max_num_prev_frames+1):(i*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_cl_list_mean[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
          std_list_data[i*(self.max_num_prev_frames+1):(i*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_cl_list_std[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
              
        time_r2_auc_img(self.max_num_prev_frames, data_list, cl_type, mean_list_data, std_list_data, len(data_list), 'auc')
      
    for data in data_list:
      
      mean_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(self.reg_list) ))
      std_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(self.reg_list) ))
      for reg_type, j in zip(self.reg_list, range(len(self.reg_list))):
        
        for ind in range(len(data_reg_names_list)):
          if (str(data)+"_"+str(reg_type)) == data_reg_names_list[ind]:
            break
        
        mean_list_data[j*(self.max_num_prev_frames+1):(j*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_reg_list_mean[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
        std_list_data[j*(self.max_num_prev_frames+1):(j*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_reg_list_std[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
            
      time_r2_auc_img(self.max_num_prev_frames, self.reg_list, data, mean_list_data, std_list_data, len(self.reg_list), 'r2')

      mean_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(self.cl_list) ))
      std_list_data = np.zeros(( (self.max_num_prev_frames+1)*len(self.cl_list) ))
      for cl_type, j in zip(self.cl_list, range(len(self.cl_list))):
        
        for ind in range(len(data_cl_names_list)):
          if (str(data)+"_"+str(cl_type)+"_CL") == data_cl_names_list[ind]:
            break

        mean_list_data[j*(self.max_num_prev_frames+1):(j*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_cl_list_mean[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
        std_list_data[j*(self.max_num_prev_frames+1):(j*(self.max_num_prev_frames+1))+(self.max_num_prev_frames+1)] = data_cl_list_std[ind*(self.max_num_prev_frames+1):ind*(self.max_num_prev_frames+1)+(self.max_num_prev_frames+1)]
          
      time_r2_auc_img(self.max_num_prev_frames, self.cl_list, data, mean_list_data, std_list_data, len(self.cl_list), 'auc')  
