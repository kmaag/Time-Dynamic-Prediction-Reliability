import numpy as np
import h5py
import os
import pickle
import subprocess

from defs_global          import SETUP
from in_and_out_functions import probs_gt_load, get_save_path_t_s_metrics_i,\
                                 t_s_metrics_load, t_s_metrics_dump

 


def t_s_metrics_concat( ):
  num_images = SETUP.NUM_IMAGES
  epsilon = SETUP.TRACKING_EPSILON
  num_reg = SETUP.TRACKING_NUMBER_REGR
  if os.path.isfile( get_save_path_t_s_metrics_i( "_all", epsilon, num_reg ) ):
    print("load time series metrics")
    metrics = t_s_metrics_load( "_all", epsilon, num_reg )
    start = pickle.load( open( SETUP.DIR_METRICS + "start" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".p", "rb" ) )
  else:
    print("concatenate time series metrics")
    metrics = t_s_metrics_load( 0, epsilon, num_reg )
    start = list([ 0, len(metrics["S"]) ])
    for i in range(1,num_images):
      m = t_s_metrics_load( i, epsilon, num_reg )
      start += [ start[-1]+len(m["S"]) ]
      for j in metrics:
        metrics[j] += m[j]
    t_s_metrics_dump( metrics, "_all", epsilon, num_reg )
    pickle.dump( start, open( SETUP.DIR_METRICS + "start" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".p", "wb" ) )   
  return metrics, start



def t_s_metrics_dataset( metrics, nclasses, all_metrics=[] ):
  
  epsilon = SETUP.TRACKING_EPSILON
  num_reg = SETUP.TRACKING_NUMBER_REGR
  class_names = []
  X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"] and "cprob" not in m ])
  if SETUP.CLASS_DTYPE == SETUP.CLASS_DTYPES[1]:
    class_names = [ "cprob"+str(i) for i in range(nclasses) if "cprob"+str(i) in metrics ]
  elif SETUP.CLASS_DTYPE == SETUP.CLASS_DTYPES[0]:
    class_names = ["class"]
  if os.path.isfile( SETUP.DIR_METRICS + "Xa" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy" ):
    print("load time series metrics (to dataset)")
    Xa = np.load(SETUP.DIR_METRICS + "Xa" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
    classes = np.load(SETUP.DIR_METRICS + "classes" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
    ya = np.load(SETUP.DIR_METRICS + "ya" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
    y0a = np.load(SETUP.DIR_METRICS + "y0a" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
  else:
    print("create time series metrics (to dataset)")
    Xa      = t_s_metrics_np( metrics, X_names    , normalize=True, all_metrics=all_metrics )
    np.save(os.path.join(SETUP.DIR_METRICS, "Xa" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), Xa)
    classes = t_s_metrics_np( metrics, class_names, normalize=True, all_metrics=all_metrics )
    np.save(os.path.join(SETUP.DIR_METRICS, "classes" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), classes)
    ya      = t_s_metrics_np( metrics, ["iou" ]   , normalize=False )
    np.save(os.path.join(SETUP.DIR_METRICS, "ya" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), ya)
    y0a     = t_s_metrics_np( metrics, ["iou0"]   , normalize=False )
    np.save(os.path.join(SETUP.DIR_METRICS, "y0a" + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), y0a)
  if SETUP.CLASS_DTYPE == SETUP.CLASS_DTYPES[0]:
    classes, class_names = classes_to_cat( classes, nclasses )
  return Xa, classes, ya, y0a, X_names, class_names  



def t_s_metrics_np( metrics, names, normalize=False, all_metrics=[] ):
  
  I = range(len(metrics['S_in']))
  M_with_zeros = np.zeros((len(I), len(names)))
  I = np.asarray(metrics['S_in']) > 0
  M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
  MM = []
  if all_metrics == []:
    MM = M.copy()
  else:
    MM = np.asarray( [ np.asarray(all_metrics[ m ])[I] for m in names ] )
  if normalize == True:
    for i in range(M.shape[0]):
      if names[i] != "class":
        M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
  M = np.squeeze(M.T)
  counter = 0
  for i in range(M_with_zeros.shape[0]):
    if I[i] == True and M_with_zeros.shape[1]>1:
      M_with_zeros[i,:] = M[counter,:]
      counter += 1
    if I[i] == True and M_with_zeros.shape[1]==1:
      M_with_zeros[i] = M[counter]
      counter += 1
  return M_with_zeros



def classes_to_cat( classes, nc = None ):

  classes = np.squeeze( np.asarray(classes) )
  if nc == None:
    nc      = np.max(classes)
  classes = label_as_onehot( classes.reshape( (classes.shape[0],1) ), nc ).reshape( (classes.shape[0], nc) )
  names   = [ "C_"+str(i) for i in range(nc) ]
  return classes, names



def t_s_dataset_concat( Xa, ya, y0a, start ):
  
  num_images = SETUP.NUM_IMAGES
  epsilon = SETUP.TRACKING_EPSILON
  num_reg = SETUP.TRACKING_NUMBER_REGR
  num_prev_frames = SETUP.NUMBER_OF_FRAMES
  
  print("Concatenate timeseries dataset")
  
  ya = np.squeeze(ya)
  y0a =np.squeeze(y0a)
  
  start_path = SETUP.DIR_PROBABILITIES + "../list_start.npy"
  start_list = np.load(start_path)
  starts = np.zeros((len(start_list)+1))
  starts[0:len(start_list)] = start_list[0:len(start_list)]
  starts[len(start_list)] = num_images
  
  max_num_prev_frames = num_prev_frames 
  
  if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
    
    if "A" in SETUP.TRAIN_TYPE:
      del_path = SETUP.DIR_AUGMENTED
      if os.path.exists( del_path ):
        del_files = sorted(os.listdir(del_path))
        for del_file in del_files:
          os.remove(del_path+del_file)
      
    real_gt_path = SETUP.DIR_PROBABILITIES + "../list_real_gt.npy"
    real_gt_list = np.load(real_gt_path)
  
    for num_gt in real_gt_list:
      for j in range(num_prev_frames+1):
        if (num_gt - j in start_list) and (j<max_num_prev_frames):
          max_num_prev_frames = j
          break
    print("number of previous frames: ", max_num_prev_frames)
    
    # real gt
    if os.path.isfile( SETUP.DIR_METRICS + "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy" ):
      print("load real gt")
      
      Xa_R = np.load(SETUP.DIR_METRICS + "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      ya_R = np.load(SETUP.DIR_METRICS + "ya_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      y0a_R = np.load(SETUP.DIR_METRICS + "y0a_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")

    else:
      print("generate real gt")
      
      Xa_R = np.zeros(( len(real_gt_list) * start[1], Xa.shape[1] * (max_num_prev_frames+1) ))
      ya_R = np.zeros(( len(real_gt_list) * start[1] ))
      y0a_R = np.zeros(( len(real_gt_list) * start[1] ))
      counter = 0
      
      for i in real_gt_list:
        tmp = np.zeros(( start[1], Xa.shape[1] * (max_num_prev_frames+1) ))
        for j in range(0,max_num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[start[i-j]:start[i-j+1]]
        
        Xa_R[start[counter]:start[counter+1],:] = tmp
        ya_R[start[counter]:start[counter+1]] = ya[start[i]:start[i+1]]
        y0a_R[start[counter]:start[counter+1]] = y0a[start[i]:start[i+1]]
        counter +=1
          
      # delete rows with only zeros in frame t
      not_del_rows_R = ~(Xa_R[:,0:Xa.shape[1]]==0).all(axis=1)
      Xa_R = Xa_R[not_del_rows_R]
      ya_R = ya_R[not_del_rows_R]
      y0a_R = y0a_R[not_del_rows_R]  
      
      np.save(os.path.join(SETUP.DIR_METRICS, "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), Xa_R)
      np.save(os.path.join(SETUP.DIR_METRICS, "ya_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), ya_R)
      np.save(os.path.join(SETUP.DIR_METRICS, "y0a_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), y0a_R)

    # pseudo gt
    if os.path.isfile( SETUP.DIR_METRICS + "Xa_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy" ):
      print("load pseudo gt")
      
      Xa_P = np.load(SETUP.DIR_METRICS + "Xa_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      ya_P = np.load(SETUP.DIR_METRICS + "ya_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      y0a_P = np.load(SETUP.DIR_METRICS + "y0a_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")

    else:
      print("generate pseudo gt")
        
      Xa_P = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)-len(real_gt_list)) * start[1], Xa.shape[1] * (max_num_prev_frames+1)))
      ya_P = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)-len(real_gt_list)) * start[1] ))
      y0a_P = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)-len(real_gt_list)) * start[1] ))
      counter = 0
      for c,k in zip(starts[0:len(start_list)], range(len(start_list))):
        for i in range(int(c+max_num_prev_frames),int(starts[k+1])):
          
          if i not in real_gt_list:
            tmp = np.zeros(( start[1], Xa.shape[1] * (max_num_prev_frames+1) ))
            for j in range(0,max_num_prev_frames+1):
              tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[start[i-j]:start[i-j+1]]
            
            Xa_P[start[counter]:start[counter+1],:] = tmp
            ya_P[start[counter]:start[counter+1]] = ya[start[i]:start[i+1]]
            y0a_P[start[counter]:start[counter+1]] = y0a[start[i]:start[i+1]]
            counter +=1
            
      # delete rows with only zeros in frame t
      not_del_rows_P = ~(Xa_P[:,0:Xa.shape[1]]==0).all(axis=1)
      Xa_P = Xa_P[not_del_rows_P]
      ya_P = ya_P[not_del_rows_P]
      y0a_P = y0a_P[not_del_rows_P]  
      
      np.save(os.path.join(SETUP.DIR_METRICS, "Xa_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), Xa_P)
      np.save(os.path.join(SETUP.DIR_METRICS, "ya_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), ya_P)
      np.save(os.path.join(SETUP.DIR_METRICS, "y0a_P" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), y0a_P)
    
  elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
  
    if os.path.isfile( SETUP.DIR_METRICS + "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy" ):
      print("load real gt")
      
      Xa_R = np.load(SETUP.DIR_METRICS + "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      ya_R = np.load(SETUP.DIR_METRICS + "ya_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
      y0a_R = np.load(SETUP.DIR_METRICS + "y0a_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg) + ".npy")
    
    else:
      print("generate real gt")
      
      Xa_R = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)) * start[1], Xa.shape[1] * (max_num_prev_frames+1)))
      ya_R = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)) * start[1] ))
      y0a_R = np.zeros(( (num_images-(len(start_list)*max_num_prev_frames)) * start[1] ))
      counter = 0
      
      for c,k in zip(starts[0:len(start_list)], range(len(start_list))):
        for i in range(int(c+max_num_prev_frames),int(starts[k+1])):
            
          tmp = np.zeros(( start[1], Xa.shape[1] * (max_num_prev_frames+1) ))
          for j in range(0,max_num_prev_frames+1):
            tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[start[i-j]:start[i-j+1]]
              
          Xa_R[start[1]*counter:start[1]*(counter+1),:] = tmp
          ya_R[start[1]*counter:start[1]*(counter+1)] = ya[start[i]:start[i+1]]
          y0a_R[start[1]*counter:start[1]*(counter+1)] = y0a[start[i]:start[i+1]]
          counter +=1
        
      # delete rows with only zeros in frame t
      not_del_rows_R = ~(Xa_R[:,0:Xa.shape[1]]==0).all(axis=1)
      Xa_R = Xa_R[not_del_rows_R]
      ya_R = ya_R[not_del_rows_R]
      y0a_R = y0a_R[not_del_rows_R]  
        
      np.save(os.path.join(SETUP.DIR_METRICS, "Xa_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), Xa_R)
      np.save(os.path.join(SETUP.DIR_METRICS, "ya_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), ya_R)
      np.save(os.path.join(SETUP.DIR_METRICS, "y0a_R" + "_npf" + str(max_num_prev_frames) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)), y0a_R)
      
    Xa_P = 0
    ya_P = 0
    y0a_P = 0
   
  print("shapes real, pseudo:", "Xa", np.shape(Xa_R), np.shape(Xa_P), "ya", np.shape(ya_R), np.shape(ya_P))
  
  return Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, max_num_prev_frames



def train_val_test_split( Xa_R, ya_R, y0a_R, Xa_P, ya_P, y0a_P, run = 0 ):
  
  num_images = SETUP.NUM_IMAGES
  np.random.seed( run )
  print("create train/val/test splitting")
  
  if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
    multi_fac_augm = SETUP.AUGMENTED_FACTOR
    multi_fac_pseudo = SETUP.PSEUDO_FACTOR
    real_gt_path = SETUP.DIR_PROBABILITIES + "../list_real_gt.npy"
    real_gt_list = np.load(real_gt_path)
    test_mask_R = np.random.rand(len(ya_R)) < float(SETUP.TEST_SPLIT)/100.0
    test_mask_P = np.random.rand(len(ya_P)) < np.count_nonzero(test_mask_R) / len(ya_P)
    Xa_test_R = Xa_R[test_mask_R]
    ya_test_R = ya_R[test_mask_R]
    y0a_test_R = y0a_R[test_mask_R] 
    Xa_test_P = Xa_P[test_mask_P]
    ya_test_P = ya_P[test_mask_P]
    y0a_test_P = y0a_P[test_mask_P] 
    Xa_R_tmp = Xa_R[np.logical_not(test_mask_R)]
    ya_R_tmp = ya_R[np.logical_not(test_mask_R)]
    y0a_R_tmp = y0a_R[np.logical_not(test_mask_R)]
    Xa_P_tmp = Xa_P[np.logical_not(test_mask_P)]
    ya_P_tmp = ya_P[np.logical_not(test_mask_P)]
    y0a_P_tmp = y0a_P[np.logical_not(test_mask_P)]
    train_mask_R = np.random.rand(len(ya_R_tmp)) < float(SETUP.TRAIN_SPLIT) / (100.0-float(SETUP.TEST_SPLIT))
    Xa_val = Xa_R_tmp[np.logical_not(train_mask_R)]
    ya_val = ya_R_tmp[np.logical_not(train_mask_R)]
    y0a_val = y0a_R_tmp[np.logical_not(train_mask_R)]
    
    if "A" in SETUP.TRAIN_TYPE:
      save_path_Xa = SETUP.DIR_AUGMENTED+"Xa_train_run"+str(run)+".npy"
      save_path_ya = SETUP.DIR_AUGMENTED+"ya_train_run"+str(run)+".npy"
      
      if not os.path.isfile( save_path_Xa ):
        print("create augmented gt")
        dump_dir  = os.path.dirname( save_path_Xa )
        if not os.path.exists( dump_dir ):
          os.makedirs( dump_dir )
        np.save(save_path_Xa, Xa_R_tmp[train_mask_R])
        np.save(save_path_ya, ya_R_tmp[train_mask_R])
        subprocess.check_call(['Rscript', 'up.R',str(run), SETUP.DIR_AUGMENTED], shell=False)
      
      print("load augmented gt")
      Xa_A = np.load(SETUP.DIR_AUGMENTED + "Xa_A_run" + str(run) + ".npy")
      ya_A = np.load(SETUP.DIR_AUGMENTED + "ya_A_run" + str(run) + ".npy")
      y0a_A = np.zeros(( len(ya_A) ))
      y0a_A[ya_A==0] = 1
        
    if SETUP.TRAIN_TYPE == "R":
      
      Xa_train = Xa_R_tmp[train_mask_R]
      ya_train = ya_R_tmp[train_mask_R]
      y0a_train = y0a_R_tmp[train_mask_R]
      
    elif SETUP.TRAIN_TYPE == "RA":
      
      augmented_mask = np.random.rand(len(ya_A)) < float(len(ya_R_tmp[train_mask_R])) / float(len(ya_A)) * multi_fac_augm
      Xa_train = np.concatenate( (Xa_R_tmp[train_mask_R], Xa_A[augmented_mask]), axis = 0)
      ya_train = np.concatenate( (ya_R_tmp[train_mask_R], ya_A[augmented_mask]), axis = 0)
      y0a_train = np.concatenate( (y0a_R_tmp[train_mask_R], y0a_A[augmented_mask]), axis = 0)
      
    elif SETUP.TRAIN_TYPE == "RAP":
      
      augmented_mask = np.random.rand(len(ya_A)) < float(len(ya_R_tmp[train_mask_R])) / float(len(ya_A)) * multi_fac_augm
      pseudo_mask = np.random.rand(len(ya_P_tmp)) < float(len(ya_R_tmp[train_mask_R])) / float(len(ya_P_tmp)) * multi_fac_pseudo
      Xa_train = np.concatenate( (Xa_R_tmp[train_mask_R], Xa_A[augmented_mask]), axis = 0)
      ya_train = np.concatenate( (ya_R_tmp[train_mask_R], ya_A[augmented_mask]), axis = 0)
      y0a_train = np.concatenate( (y0a_R_tmp[train_mask_R], y0a_A[augmented_mask]), axis = 0)
      Xa_train = np.concatenate( (Xa_train, Xa_P_tmp[pseudo_mask]), axis = 0)
      ya_train = np.concatenate( (ya_train, ya_P_tmp[pseudo_mask]), axis = 0)
      y0a_train = np.concatenate( (y0a_train, y0a_P_tmp[pseudo_mask]), axis = 0)
      
    elif SETUP.TRAIN_TYPE == "RP":
      
      pseudo_mask = np.random.rand(len(ya_P_tmp)) < float(len(ya_R_tmp[train_mask_R])) / float(len(ya_P_tmp)) * multi_fac_pseudo
      Xa_train = np.concatenate( (Xa_R_tmp[train_mask_R], Xa_P_tmp[pseudo_mask]), axis = 0)
      ya_train = np.concatenate( (ya_R_tmp[train_mask_R], ya_P_tmp[pseudo_mask]), axis = 0)
      y0a_train = np.concatenate( (y0a_R_tmp[train_mask_R], y0a_P_tmp[pseudo_mask]), axis = 0)
    
    elif SETUP.TRAIN_TYPE == "P":
      
      pseudo_mask = np.random.rand(len(ya_P_tmp)) < float(len(ya_R_tmp[train_mask_R])) / float(len(ya_P_tmp)) * multi_fac_pseudo
      Xa_train = Xa_P_tmp[pseudo_mask]
      ya_train = ya_P_tmp[pseudo_mask]
      y0a_train = y0a_P_tmp[pseudo_mask]
     
  elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:

    minimize_mask_R = np.random.rand(len(ya_R)) < 38000.0 / float(len(ya_R)) 
    Xa_R_tmp = Xa_R[minimize_mask_R]
    ya_R_tmp = ya_R[minimize_mask_R]
    y0a_R_tmp = y0a_R[minimize_mask_R] 
    train_mask_R = np.random.rand(len(ya_R_tmp)) < float(SETUP.TRAIN_SPLIT) / 100.0
    Xa_train = Xa_R_tmp[train_mask_R]
    ya_train = ya_R_tmp[train_mask_R]
    y0a_train = y0a_R_tmp[train_mask_R]
    Xa_R_tmp1 = Xa_R_tmp[np.logical_not(train_mask_R)]
    ya_R_tmp1 = ya_R_tmp[np.logical_not(train_mask_R)]
    y0a_R_tmp1 = y0a_R_tmp[np.logical_not(train_mask_R)]
    val_mask_R = np.random.rand(len(ya_R_tmp1)) < float(SETUP.VAL_SPLIT) / (100.0-float(SETUP.TRAIN_SPLIT))
    Xa_val = Xa_R_tmp1[val_mask_R]
    ya_val = ya_R_tmp1[val_mask_R]
    y0a_val = y0a_R_tmp1[val_mask_R]
    Xa_test_R = Xa_R_tmp1[np.logical_not(val_mask_R)]
    ya_test_R = ya_R_tmp1[np.logical_not(val_mask_R)]
    y0a_test_R = y0a_R_tmp1[np.logical_not(val_mask_R)]
    Xa_test_P = Xa_test_R.copy()
    ya_test_P = ya_test_R.copy()
    y0a_test_P = y0a_test_R.copy()
  ya_train = np.squeeze(ya_train)
  y0a_train =np.squeeze(y0a_train)
  ya_val = np.squeeze(ya_val)
  y0a_val =np.squeeze(y0a_val)
  ya_test_R = np.squeeze(ya_test_R)
  y0a_test_R =np.squeeze(y0a_test_R)
  ya_test_P = np.squeeze(ya_test_P)
  y0a_test_P =np.squeeze(y0a_test_P)
  
  return Xa_train, Xa_val, Xa_test_R, Xa_test_P, ya_train, ya_val, ya_test_R, ya_test_P, y0a_train, y0a_val, y0a_test_R, y0a_test_P
  
  

def gen_list_starts_gt( ):
  
  print("generate list with starts")
  save_path = SETUP.DIR_PROBABILITIES + "../"
  num_images = SETUP.NUM_IMAGES
  
  if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:

    start_list = []
    real_gt_list = []
    
    for i in range(num_images):
      probs, gt, filename = probs_gt_load( i )
      filename = filename[27:37]
      if int(filename) == 0:
        start_list.append( (i) )  
      gt_xception_path = SETUP.DIR_PROBABILITIES + "../xc.mscl.os8/" + "probs_" + str(i) +".hdf5"
      f_probs = h5py.File( gt_xception_path , "r")
      gt_xception = np.asarray( f_probs['ground_truths'] )
      gt_xception = np.squeeze( gt_xception[0] )
      if np.sum(gt_xception) > 0:
        real_gt_list.append( i )
    
    print(start_list)
    print(real_gt_list)
    np.save(os.path.join(save_path, "list_start"), start_list)
    np.save(os.path.join(save_path, "list_real_gt"), real_gt_list)
    
  elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
    
    start_list = []
    counter=1
    
    for i in range(num_images):
      _, _, filename = probs_gt_load( i )
      filename = filename[4:9]
      if int(filename) == 1:
        start_list.append( (i) ) 
        counter=1
      if counter!=int(filename):
        break
      counter+=1
    
    print(start_list)
    np.save(os.path.join(save_path, "list_start"), start_list) 
    
    

def name_to_latex( name ):
  
  for i in range(100):
    if name == "cprob"+str(i):
      return "$C_{"+str(i)+"}$"

  mapping = {'E': '$\\bar E$',
             'E_bd': '${\\bar E}_{bd}$',
             'E_in': '${\\bar E}_{in}$',
             'E_rel_in': '$\\tilde{\\bar E}_{in}$',
             'E_rel': '$\\tilde{\\bar E}$',
             'M': '$\\bar M$',
             'M_bd': '${\\bar M}_{bd}$',
             'M_in': '${\\bar M}_{in}$',
             'M_rel_in': '$\\tilde{\\bar M}_{in}$',
             'M_rel': '$\\tilde{\\bar M}$',
             'S': '$S$',
             'S_bd': '${S}_{bd}$',
             'S_in': '${S}_{in}$',
             'S_rel_in': '$\\tilde{S}_{in}$',
             'S_rel': '$\\tilde{S}$',
             'V': '$\\bar V$',
             'V_bd': '${\\bar V}_{bd}$',
             'V_in': '${\\bar V}_{in}$',
             'V_rel_in': '$\\tilde{\\bar V}_{in}$',
             'V_rel': '$\\tilde{\\bar V}$',
             'mean_x' : '${\\bar x}$',
             'mean_y' : '${\\bar y}$', 
             'C_p' : '${C}_{p}$',
             'iou' : '$IoU_{adj}$'}        
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name)
