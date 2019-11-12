import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
plt.rc('font', size=10, family='serif')
plt.rc("text", usetex=True)

import label_file as label_file
from defs_global          import SETUP
from in_and_out_functions import probs_gt_load, t_s_con_comp_load, get_img_path_fname,\
                                 t_s_metrics_load, get_save_path_probs_i

if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
  trainId2label = { label.trainId : label for label in reversed(label_file.kitti_labels) }
  
elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
  trainId2label = { label.trainId : label for label in reversed(label_file.gta_labels) }



def time_r2_auc_img( max_num_prev_frames, names, data_type, mean_list_data, std_list_data, number, r2_or_auc):
  
  size_font = 14
  num_timeseries = np.arange(1, max_num_prev_frames+2)
  if number == 5:
    color_map = ['slateblue', 'indianred', 'mediumseagreen', 'darkgoldenrod', 'mediumvioletred']
  elif number == 6: 
    color_map = ['cornflowerblue', 'mediumorchid', 'darkseagreen', 'lightskyblue', 'mediumslateblue', 'plum']
  elif number == 3:
    color_map = ['mediumorchid', 'lightskyblue', 'mediumslateblue']
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  for i in range(number):
    if number == 5:
      name_tmp = names[i]
      name_label = name_tmp[0:len(name_tmp)-8]
    elif number == 6 or number == 3:
      name_tmp = names[i]
      name_label = name_tmp.replace("_", " ")
    r_min = (max_num_prev_frames+1) * i
    r_max = (max_num_prev_frames+1) * (i+1)
    plt.plot(num_timeseries, mean_list_data[r_min:r_max], color=color_map[i], marker='o', label=name_label)  
    plt.fill_between(num_timeseries, mean_list_data[r_min:r_max]-std_list_data[r_min:r_max], mean_list_data[r_min:r_max]+std_list_data[r_min:r_max], color=color_map[i], alpha=0.05 ) 
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.xlabel('number of considered frames', fontsize=size_font)
  if r2_or_auc == 'auc':
    plt.ylabel('$AUROC$', fontsize=size_font)
  else:
    plt.ylabel('$R^2$', fontsize=size_font)
  if number == 5:
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., prop={'size': 10})
  else: 
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0., prop={'size': 10})
  save_path = SETUP.DIR_IMAGE_ANALYZE + "all_" + data_type + "_" + str(r2_or_auc) + "_timeline.pdf"   
  f1.savefig(save_path)#, dpi=300)
  plt.close()
  
  

def timeline_img( num_timeseries, train, train_std, val, val_std, test_R, test_R_std, test_P, test_P_std, analyze_type ):
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(num_timeseries, train, color='violet', marker='o', label="train")  
  plt.fill_between(num_timeseries, train-train_std, train+train_std, color='violet', alpha=0.05 )
  plt.plot(num_timeseries, val, color='midnightblue', marker='o', label="val")
  plt.fill_between(num_timeseries, val-val_std, val+val_std, color='midnightblue', alpha=0.05 )
  plt.plot(num_timeseries, test_R, color='deepskyblue', marker='o', label="test real")  
  plt.fill_between(num_timeseries, test_R-test_R_std, test_R+test_R_std, color='deepskyblue', alpha=0.05 )
  if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
    plt.plot(num_timeseries, test_P, color='c', marker='o', label="test pseudo")  
    plt.fill_between(num_timeseries, test_P-test_P_std, test_P+test_P_std, color='c', alpha=0.05 )
  plt.xlabel('Frames')
  if analyze_type == 'r2':
    plt.ylabel('$R^2$')
    name = SETUP.REGRESSION_TYPE + "_r2_timeline.png"
  elif analyze_type == 'auc':
    plt.ylabel('$AUROC$')
    name = SETUP.CLASSIFICATION_TYPE + "_auc_timeline.png"
  elif analyze_type == 'acc':
    plt.ylabel('$ACC$')
    name = SETUP.CLASSIFICATION_TYPE + "_acc_timeline.png"
  if SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[0]:
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
  elif SETUP.IMAGE_TYPE == SETUP.IMAGE_TYPES[1]:
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
  save_path = SETUP.DIR_ANALYZE + name
  f1.savefig(save_path, dpi=300)
  plt.close()


  
def plot_regression_scatter( Xa_test, ya_test, ya_test_pred, X_names, num_frames ):
  
  #os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  cmap=plt.get_cmap('tab20')

  S_ind = 0
  for S_ind in range(len(X_names)):
    if X_names[S_ind] == "S":
      break
  
  figsize=(3.0,13.0/5.0)
  plt.figure(figsize=figsize, dpi=300)
  plt.clf()
  
  sizes = np.squeeze(Xa_test[:,S_ind]*np.std(Xa_test[:,S_ind]))
  sizes = sizes - np.min(sizes)
  sizes = sizes / np.max(sizes) * 50 #+ 1.5      
  x = np.arange(0., 1, .01)
  plt.plot( x, x, color='black' , alpha=0.5, linestyle='dashed')
  plt.scatter( ya_test, np.clip(ya_test_pred,0,1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
  plt.ylabel('predicted $\mathit{IoU}_\mathrm{adj}$')
  plt.savefig(SETUP.DIR_ANALYZE+'scatter/' + SETUP.REGRESSION_TYPE + '_scatter_test_npf' + str(num_frames) + '.png', bbox_inches='tight')
  plt.close()
  
  

def t_s_metrics_i_img( iou, iou_pred, i, name=" " ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    probs, gt, filename = probs_gt_load( i )
    path = get_img_path_fname( filename )
    input_image = np.asarray(Image.open( path ))
    
    con_comp = t_s_con_comp_load( i, SETUP.TRACKING_EPSILON, SETUP.TRACKING_NUMBER_REGR)
    con_comp[ gt == 255 ] = 0
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype='int' )
    gt[ gt == 255 ] = 0   
    
    predc = np.asarray([ trainId2label[ pred[p,q] ].color for p in range(pred.shape[0]) for q in range(pred.shape[1]) ])
    gtc   = np.asarray([ trainId2label[ gt[p,q]   ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
    predc = predc.reshape(input_image.shape)
    gtc   = gtc.reshape(input_image.shape)
    
    img_iou = visualize_segments( con_comp, iou )
    
    I4 = predc / 2.0 + input_image / 2.0 
    I3 = gtc / 2.0 + input_image / 2.0
    
    img_pred = visualize_segments( con_comp, iou_pred )

    img = np.concatenate( (img_iou,img_pred), axis=1 )
    img2 = np.concatenate( (I3,I4), axis=1 )
    img = np.concatenate( (img,img2), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    
    seg_dir = SETUP.DIR_IMAGES_REGRESSION
    if not os.path.exists( seg_dir ):
      os.makedirs( seg_dir )
    image.save(seg_dir+name+str(i).zfill(10)+".png")
    print("stored:",seg_dir+name+str(i).zfill(10)+".png")
  
  
  
def scatter_lifetime_plot( mean_lifetime, lifetime_mean_size_del, size_cut, lifetime_mean_size_del_cut ):
  
  save_path = SETUP.DIR_IMAGES_SCATTER
  if not os.path.exists( save_path ):
    os.makedirs( save_path )                    
  x = np.arange(2)
  f2, ax = plt.subplots()
  plt.bar(x, mean_lifetime, color='teal')
  plt.xticks(x, ('mean lifetime', 'mean lifetime for segments with mean $S_{in} >$' + str(size_cut))) 
  plt.ylabel('frames')
  plt.title("Mean lifetime for segments")
  f2.savefig(save_path+"mean_lifetime.png", dpi=300, bbox_inches='tight')
  plt.close()
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  figsize=(3.0,13.0/5.0)
  plt.figure(figsize=figsize, dpi=300)
  plt.scatter( lifetime_mean_size_del[:,0], lifetime_mean_size_del[:,1], s = 10, c='palevioletred', alpha=0.05 )
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel("lifetime")
  plt.ylabel("mean $S_{in}$")
  plt.savefig(save_path+'lifetime_size_loglog.png', bbox_inches='tight')
  plt.close()



def plot_matching( n, colors_list ):
  
  t = time.time()
  
  save_path = SETUP.DIR_IMAGES_T_S
  
  epsilon = SETUP.TRACKING_EPSILON
  num_reg = SETUP.TRACKING_NUMBER_REGR
  probs, gt, filename  = probs_gt_load( n )
  mask_orig = probs.argmax(axis=-1)
  t_s_comp_before = t_s_con_comp_load( n - 1, epsilon, num_reg )
  t_s_comp = t_s_con_comp_load( n, epsilon, num_reg )
  t_s_comp_after = t_s_con_comp_load( n + 1, epsilon, num_reg )
  path = get_img_path_fname(filename)
  input_image = np.asarray( Image.open(path) )
 
  mask_orig_2= np.asarray([ trainId2label[ mask_orig[p,q] ].color for p in range(mask_orig.shape[0]) for q in range(mask_orig.shape[1]) ])
  mask_orig_2 = mask_orig_2.reshape(input_image.shape)
  # I1 mask image n
  I1 = mask_orig_2 / 2.0 + input_image / 2.0
      
  len_colors_list = len(colors_list)
  colors_t = np.zeros((input_image.shape[0], input_image.shape[1], input_image.shape[2]))
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if t_s_comp[i,j] < 0:
        colors_t[i,j,:] = (0,0,0)
      else:
        colors_t[i,j,:] = hex_to_rgb( colors_list[ (t_s_comp[i,j]-1) % len_colors_list ] )
  I2 = colors_t
  
  gt_color= np.asarray([ trainId2label[ gt[p,q] ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
  gt_color = gt_color.reshape(input_image.shape)
  I3 = gt_color / 2.0 + input_image / 2.0
  
  new_con_comp = np.zeros((input_image.shape[0], input_image.shape[1], input_image.shape[2]))
  lost_con_comp = np.zeros((input_image.shape[0], input_image.shape[1], input_image.shape[2]))
  ind_new_in = t_s_comp[:,:] > -t_s_comp_before[:,:].min() 
  ind_new_bd = t_s_comp[:,:] < t_s_comp_before[:,:].min()
  ind_lost_in = np.zeros((input_image.shape[0], input_image.shape[1]))
  ind_lost_bd = np.zeros((input_image.shape[0], input_image.shape[1]))
  for k in range(1, -t_s_comp[:,:].min() +1 ):
    if ( np.count_nonzero(t_s_comp_after[:,:]==k) + np.count_nonzero(t_s_comp_after[:,:]==-k) ) == 0:
      tmp_in =  t_s_comp[:,:] == k
      tmp_bd =  t_s_comp[:,:] == -k
      ind_lost_in = ind_lost_in + tmp_in
      ind_lost_bd = ind_lost_bd + tmp_bd
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if ind_new_in[i,j] == 1:
        new_con_comp[i,j,0] = 239
        new_con_comp[i,j,1] = 79
        new_con_comp[i,j,2] = 117
      if ind_new_bd[i,j] == 1:
        new_con_comp[i,j,0] = 255
        new_con_comp[i,j,1] = 255
        new_con_comp[i,j,2] = 255
      if ind_lost_in[i,j] == 1:
        lost_con_comp[i,j,0] = 75
        lost_con_comp[i,j,1] = 165
        lost_con_comp[i,j,2] = 187
      if ind_lost_bd[i,j] == 1:
        lost_con_comp[i,j,0] = 255
        lost_con_comp[i,j,1] = 255
        lost_con_comp[i,j,2] = 255    
  I4 = lost_con_comp + new_con_comp
      
  img12 = np.concatenate( (I1,I2), axis=1 )
  img34 = np.concatenate( (I3,I4), axis=1 )
  img = np.concatenate( (img12,img34), axis=0 )
  image = Image.fromarray(img.astype('uint8'), 'RGB')
  image.save(save_path + "img_ts_comp" + str(n).zfill(10) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)  + ".png")
  
  img_small = I2 * 0.7 + input_image * 0.3
  image_color_comp_inp = Image.fromarray(img_small.astype('uint8'), 'RGB')
  image_color_comp_inp.save(save_path + "mini_img_ts_comp" + str(n).zfill(10) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)  + ".png")
  
  print("plot image", n, ": time needed ", time.time()-t)  
  


def visualize_segments( comp, metric ):
  
  R = np.asarray( metric )
  R = 1-0.5*R
  G = np.asarray( metric )
  B = 0.3+0.35*np.asarray( metric )
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  con_comp = np.asarray(comp.copy(), dtype='int16')
  con_comp[con_comp  < 0] = len(R)-1  
  con_comp[con_comp == 0] = len(R)    
  img = np.zeros( con_comp.shape+(3,) )
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      img[x,y,0] = R[con_comp[x,y]-1]
      img[x,y,1] = G[con_comp[x,y]-1]
      img[x,y,2] = B[con_comp[x,y]-1]
  img = np.asarray( 255*img ).astype('uint8')
  
  return img



def hex_to_rgb(input1):
  value1 = input1.lstrip('#')
  return tuple(int(value1[i:i+2], 16) for i in (0, 2 ,4))
