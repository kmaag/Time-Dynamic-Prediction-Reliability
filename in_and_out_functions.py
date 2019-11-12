import numpy as np
import h5py
import os
import pickle

from defs_global import SETUP



def get_img_path_fname( filename ):
  
  path = []
  for root, dirnames, filenames in os.walk(SETUP.DIR_IMAGES):
    for fn in filenames:
      if filename in fn:
        path = os.path.join(root, fn)
        break
  if path == []:
    print("file", filename, "not found.")
    
  return path



def get_save_path_probs_i( i ):
  
  return SETUP.DIR_PROBABILITIES + "probs_" + str(i) +".hdf5"



def get_save_path_con_comp_i( i ):
  
  return SETUP.DIR_CON_COMPONENTS + "components" + str(i).zfill(10) +".p"



def get_save_path_t_s_con_comp_i( i, eps, num_reg ):
  
  return SETUP.DIR_COMPONENTS_T_S + "t_s_components" + str(i).zfill(10) + "_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"



def get_save_path_t_s_metrics_i( i, eps, num_reg ):
  
  return SETUP.DIR_METRICS + "t_s_metrics" + str(i).zfill(10) + "_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"



def probs_gt_load( i ):
  
  f_probs = h5py.File( get_save_path_probs_i(i) , "r")
  probs   = np.asarray( f_probs['probabilities'] )
  gt      = np.asarray( f_probs['ground_truths'] )
  probs   = np.squeeze( probs )
  gt      = np.squeeze( gt[0] )
  
  return probs, gt, f_probs['file_names'][0].decode('utf8') 



def con_comp_dump( con_comp, i ):
  
  dump_path = get_save_path_con_comp_i( i )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( con_comp, open( dump_path, "wb" ) )
  
  
  
def con_comp_load( i ):
  
  read_path = get_save_path_con_comp_i( i )
  con_comp = pickle.load( open( read_path, "rb" ) )
  
  return con_comp



def t_s_con_comp_dump( t_s_con_comp, i, eps, num_reg ):
  
  dump_path = get_save_path_t_s_con_comp_i( i, eps, num_reg )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( t_s_con_comp, open( dump_path, "wb" ) )
  


def t_s_con_comp_load( i, eps, num_reg ):
  
  read_path = get_save_path_t_s_con_comp_i( i, eps, num_reg )
  t_s_con_comp = pickle.load( open( read_path, "rb" ) )
  
  return t_s_con_comp



def t_s_metrics_dump( t_s_metrics, i, eps, num_reg ):

  dump_path = get_save_path_t_s_metrics_i( i, eps, num_reg )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( t_s_metrics, open( dump_path, "wb" ) )



def t_s_metrics_load( i, eps, num_reg ):
  
  read_path = get_save_path_t_s_metrics_i( i, eps, num_reg )
  t_s_metrics = pickle.load( open( read_path, "rb" ) )
  
  return t_s_metrics


