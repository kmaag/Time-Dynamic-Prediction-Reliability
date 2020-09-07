import numpy as np
cimport numpy as np 
import os
from scipy.stats import linregress
from skimage import measure as ms
 
from defs_global          import SETUP
from in_and_out_functions import con_comp_load, get_save_path_t_s_con_comp_i,\
                          probs_gt_load, t_s_con_comp_load, t_s_con_comp_dump
                          


 
def build_con_comp( int i, int j, unsigned char[:,:] seg, short int seg_ind, np.ndarray marked_array, np.ndarray flag_array,
                    float[:,:,:] probs, unsigned char[:,:] gt,
                    unsigned short int[:,:] members_k, unsigned short int[:,:] members_l,
                    int nclasses, int x_max, int y_max ):
  
  cdef int k, l, ii, jj, x, y, n_in, n_bd, c, I, U, flag_max_x, flag_min_x, flag_max_y, flag_min_y, ic
  cdef unsigned char[:,:] flag
  cdef short int[:,:] marked
  
  if seg[i,j] < 255:
    c = seg[i,j]
    members_k[0,0], members_k[0,1] = i, j
    marked = marked_array
    flag_min_x = flag_max_x = i
    flag_min_y = flag_max_y = j
    flag = flag_array
    flag[i,j] = 1
    I = 0
    marked[i,j] = seg_ind
    k = 1
    l = 0
    while k > 0 or l > 0:
      flag_k = 0
      if k > 0:
        k -= 1
        x, y = members_k[k]
        flag_k = 1
      elif l > 0:
        l -= 1
        x, y = members_l[l]
      if flag_k:
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            if seg[ii,jj] == c and marked[ii,jj] == 0:
                marked[ii,jj] = seg_ind
                flag[ii,jj] = 1
                if ii > flag_max_x:
                  flag_max_x = ii
                elif ii < flag_min_x:
                  flag_min_x = ii
                if jj > flag_max_y:
                  flag_max_y = jj
                elif jj < flag_min_y:
                  flag_min_y = jj
                members_k[k,0], members_k[k,1] = ii, jj
                k += 1
            elif seg[ii,jj] != c:
                marked[x,y] = -seg_ind
                if gt != []:
                  if gt[ii,jj] == c and flag[ii,jj]==0:
                    flag[ii,jj] = 1
                    if ii > flag_max_x:
                      flag_max_x = ii
                    elif ii < flag_min_x:
                      flag_min_x = ii
                    if jj > flag_max_y:
                      flag_max_y = jj
                    elif jj < flag_min_y:
                      flag_min_y = jj
                    members_l[l,0], members_l[l,1] = ii, jj
                    l += 1
      if not flag_k and gt != []:
        if I == 0:
          break
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            if gt[ii,jj] == c and flag[ii,jj]==0 and seg[ii,jj] != c:
              flag[ii,jj] = 1
              if ii > flag_max_x:
                flag_max_x = ii
              elif ii < flag_min_x:
                flag_min_x = ii
              if jj > flag_max_y:
                flag_max_y = jj
              elif jj < flag_min_y:
                flag_min_y = jj
              members_l[l,0], members_l[l,1] = ii, jj
              l += 1
      if flag_k:
        if marked[x,y] in [seg_ind,-seg_ind]:
          if gt != []:
            if gt[x,y] == c:
              I += 1 
    for ii in range(flag_min_x,flag_max_x+1):
      for jj in range(flag_min_y,flag_max_y+1):
        flag[ii,jj] = 0
    seg_ind +=1
      
  return marked_array, seg_ind



def compute_con_comp( probs, gt ):
  
  cdef int i, j
  cdef short int seg_ind
  cdef np.ndarray marked
  cdef np.ndarray members_k
  cdef np.ndarray members_l
  cdef short int[:,:] M
  
  nclasses  = probs.shape[-1]
  dims      = np.asarray( probs.shape[:-1], dtype="uint16" )
  gt        = np.asarray( gt, dtype="uint8" )
  probs     = np.asarray( probs, dtype="float32" )
  seg       = np.asarray( prediction(probs, gt, ignore=False ), dtype="uint8" )
  marked    = np.zeros( dims, dtype="int16" )
  members_k = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  members_l = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  flag      = np.zeros( dims, dtype="uint8" )
  M         = marked
  
  seg_ind = 1
  for i in range(dims[0]):
    for j in range(dims[1]):
      if M[i,j] == 0:
        marked, seg_ind = build_con_comp( i, j, seg, seg_ind, marked, flag, probs, gt, members_k, members_l, nclasses, dims[0], dims[1] )
  return marked



def prediction(probs, gt, ignore=True ):
  pred = np.asarray( np.argmax( probs, axis=-1 ), dtype="uint8" )
  if ignore == True:
    pred[ gt==255 ] = 255
  return pred



def comp_t_s_con_comp_per_video(start_image, stop_image): 
  
  cdef int epsilon, num_reg, reg_steps, imx, imy, n, m, i, j, max_index
  cdef float percentage, max_components_number, counter
  cdef np.ndarray t_s_con_comp_in
  cdef np.ndarray con_comp_n_in
  cdef np.ndarray seg_in
  cdef np.ndarray flag_comp_num_n_in
  cdef np.ndarray flag_comp_num_n_1_in
  cdef short int[:,:,:] t_s_con_comp
  cdef short int[:,:] con_comp_n
  cdef short int[:] flag_comp_num_n_1
  cdef short int[:] flag_comp_num_n
  cdef float[:] inx_field_counter

  epsilon = SETUP.TRACKING_EPSILON
  num_reg = SETUP.TRACKING_NUMBER_REGR
  eps_near = epsilon/10
  eps_time = epsilon/2
  percentage = 0.35
  con_comp_0  = con_comp_load( 0 )
  con_comp_0 = np.asarray( con_comp_0, dtype="int16" )
  imx = con_comp_0.shape[0]
  imy = con_comp_0.shape[1]
  max_components_number = 0
  
  for n in range(start_image, stop_image+1):
    
    if os.path.isfile( get_save_path_t_s_con_comp_i( n, epsilon, num_reg ) ):  
      print("skip image", n)
      
      time_series_components_tmp = t_s_con_comp_load( n, epsilon, num_reg )
      max_components_number = max(max_components_number, -time_series_components_tmp.min())
      
    else:
      
      t_s_con_comp_in = np.zeros((num_reg+1,imx, imy), dtype="int16")
      t_s_con_comp = t_s_con_comp_in
      seg_in = np.zeros((num_reg+1,imx, imy), dtype="uint8")
      con_comp_n_in  = con_comp_load( n )
      con_comp_n_in = np.asarray( con_comp_n_in, dtype="int16" )
      con_comp_n = con_comp_n_in
      probs_n, gt, _ = probs_gt_load( n )
      probs_n = np.asarray( probs_n, dtype="float32" )
      gt = np.asarray( gt, dtype="uint8" )
      seg_in[0,:,:] = np.asarray( prediction(probs_n, gt, ignore=False ), dtype="uint8" )
            
      for m in range(1,num_reg+1):
        
        if n >= start_image + m: 
          
          t_s_con_comp_in[m,:,:] = t_s_con_comp_load( n-m, epsilon, num_reg )
          probs_tmp, gt, _ = probs_gt_load( n-m )
          probs_tmp = np.asarray( probs_tmp, dtype="float32" )
          gt = np.asarray( gt, dtype="uint8" )
          seg_in[m,:,:] = np.asarray( prediction(probs_tmp, gt, ignore=False ), dtype="uint8" )
        
        
      #### nearest neighbour matching:  
      print("start nearest neighbour matching")
      
      con_comp_n_in, flag_comp_num_n_in = step1(con_comp_n_in, seg_in, eps_near)
      flag_comp_num_n = flag_comp_num_n_in           

      # specialcase image 0
      if n == start_image:
        t_s_con_comp_in[0,:,:] = con_comp_n_in.copy()
      
      else:
      
        flag_comp_num_n_1_in = np.zeros( (-int(t_s_con_comp_in[1,:,:].min())), dtype="int16" )
        flag_comp_num_n_1 = flag_comp_num_n_1_in
      
        inx_field_counter = np.zeros((-int(t_s_con_comp_in[1,:,:].min())), dtype="float32" )
        inx_field = np.zeros((-int(t_s_con_comp_in[1,:,:].min())), dtype="int16" )
        for k in range( 1, -int(t_s_con_comp_in[1,:,:].min())+1 ):
          inx_field_counter[k-1] = ( np.count_nonzero(t_s_con_comp_in[1,:,:]==k) + np.count_nonzero(t_s_con_comp_in[1,:,:]==-k) )
      
        for k in range(-int(t_s_con_comp_in[1,:,:].min()) ):
          max_index = int(np.argmax(inx_field_counter))
          inx_field[k] = max_index + 1
          inx_field_counter[max_index] = -1
          
        for i in inx_field:
          if np.count_nonzero(t_s_con_comp_in[1,:,:]==-i) == 0:
            flag_comp_num_n_1[i-1] = 1
        
        #### geometric center matching
        print("start geometric center matching")
        # loop for geometric center matching
        for i in inx_field:
          
          ## min three images and the component i (n-2) has to be existent
          if n >= start_image+2 and np.count_nonzero(t_s_con_comp_in[2,:,:]==-i) > 0:
              
             t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = step2_1(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, percentage)
             t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = step2_2(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, epsilon)
            
          else:
            
            t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = simplified_step2_2(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, epsilon)
    
        
        #### overlapping matching:
        print("start overlapping matching")
        for i in inx_field:
          
          t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = step3(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, percentage)
        
        if n >= start_image + 3:
          #### time series matching
          print("start time series matching")
          if n < start_image + num_reg:
            reg_steps = n - start_image
          else:
            reg_steps = num_reg
                
          # loop for time series matching
          for i in inx_field:
            
            t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = step4_1(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, eps_time, reg_steps)
            t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in = step4_2(i, imx, imy, flag_comp_num_n_1_in, flag_comp_num_n_in, t_s_con_comp_in, seg_in, con_comp_n_in, percentage, reg_steps)
            
        #### remaining numbers
        print("start remaining numbers")  
        max_components_number = max(max_components_number, -t_s_con_comp_in[1,:,:].min())
        counter = max_components_number
        for j in range( 1, -int(con_comp_n_in.min())+1 ):
          if flag_comp_num_n[j-1] == 0:
            print("new number: number, index j, class", counter+1, j)
            t_s_con_comp_in[ 0,con_comp_n_in==j ] = counter+1
            t_s_con_comp_in[ 0,con_comp_n_in==-j ] = -(counter+1)
            flag_comp_num_n[j-1] = 1
            counter = counter + 1
      
      t_s_con_comp_dump( t_s_con_comp_in[0,:,:], n, epsilon, num_reg )   
      print("finished image", n)   
      


def step1(np.ndarray con_comp_n_in, np.ndarray seg_in, int eps_near):
  
  cdef int k, max_index, j1, j2, flag, c1, c2
  cdef float dist
  cdef float[:] inx_field_counter
  cdef long int[:,:] bound1
  cdef long int[:,:] bound2
  
  inx_field_counter = np.zeros((-int(con_comp_n_in[:,:].min())), dtype="float32" )
  inx_field = np.zeros((-int(con_comp_n_in[:,:].min())), dtype="int16" )
  for k in range( 1, -int(con_comp_n_in[:,:].min())+1 ):
    inx_field_counter[k-1] = ( np.count_nonzero(con_comp_n_in[:,:]==k) + np.count_nonzero(con_comp_n_in[:,:]==-k) )

  for k in range(-int(con_comp_n_in[:,:].min()) ):
    max_index = int(np.argmax(inx_field_counter))
    inx_field[k] = max_index + 1
    inx_field_counter[max_index] = -1
      
  flag_comp_num_n_in = np.zeros( (-int(con_comp_n_in.min())), dtype="int16" )
  flag_comp_num_n = flag_comp_num_n_in
  for j1 in inx_field:
    if flag_comp_num_n[j1-1] == 0:
      ind_bd1 = con_comp_n_in == -j1
      class_n1 = int(seg_in[0,ind_bd1].min())
      for j2 in inx_field:
        if flag_comp_num_n[j2-1] == 0 and j1 != j2:
          ind_bd2 = con_comp_n_in == -j2
          class_n2 = int(seg_in[0,ind_bd2].min())
          if class_n1 == class_n2:
            bound1_in = np.column_stack(np.where(ind_bd1))
            bound1 = bound1_in
            bound2_in = np.column_stack(np.where(ind_bd2))
            bound2 = bound2_in
            flag = 0
            c1 = 0
            while (flag == 0) and (c1 < len(bound1_in)):
              c2 = 0
              while (flag == 0) and (c2 < len(bound2_in)):
                dist = ( (bound2[c2,0] - bound1[c1,0])**2 + (bound2[c2,1] - bound1[c1,1])**2 )**0.5
                if dist <= eps_near:
                  con_comp_n_in[ con_comp_n_in==j2 ] = j1
                  con_comp_n_in[ con_comp_n_in==-j2 ] = -j1
                  flag_comp_num_n[j2-1] = 1
                  print("nearest neighbour match: index j1 and j2, class", j1, j2, class_n1)
                  flag = 1
                c2 += 1
              c1 += 1
              
  return con_comp_n_in, flag_comp_num_n_in



def step2_1(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, float percentage):
  
  cdef int x, y, class_n_1, class_n, counter, x_shift, y_shift, x_t, y_t, max_index
  cdef float mean_x_n_1, mean_y_n_1, mean_x_n_2, mean_y_n_2, number_pixel_j, max_union
  cdef short int[:,:] shifted_comp
  
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in
  
  if flag_comp_num_n_1[i-1] == 0:
    ind_n_1 = t_s_con_comp_in[1,:,:] == -i
    class_n_1 = int(seg_in[1,ind_n_1].min())
    mean_x_n_1 = 0
    mean_y_n_1 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[1,x,y] == i or t_s_con_comp[1,x,y] == -i:
          mean_x_n_1 += x
          mean_y_n_1 += y
          counter = counter + 1
    mean_x_n_1 /= counter
    mean_y_n_1 /= counter       
    mean_x_n_2 = 0
    mean_y_n_2 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[2,x,y] == i or t_s_con_comp[2,x,y] == -i:
          mean_x_n_2 += x
          mean_y_n_2 += y
          counter = counter + 1
    mean_x_n_2 /= counter
    mean_y_n_2 /= counter
    x_shift = int(mean_x_n_1 - mean_x_n_2)
    y_shift = int(mean_y_n_1 - mean_y_n_2)
    shifted_comp_in = np.zeros((imx, imy), dtype="int16")
    shifted_comp = shifted_comp_in
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[1,x,y] == i or t_s_con_comp[1,x,y] == -i:
          x_t = x + x_shift
          y_t = y + y_shift
          if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
            shifted_comp[x_t,y_t] = i
    max_union = 0
    max_index = 0
    for j in range( 1, -int(con_comp_n_in.min())+1 ):
      if flag_comp_num_n[j-1] == 0:
        ind_n = con_comp_n_in == -j
        class_n = int(seg_in[0,ind_n].min())
        if class_n_1 == class_n:
          counter = 0
          for x in range(imx):
            for y in range(imy):
              if ( shifted_comp[x,y] == i ) and ( con_comp_n[x,y] == j or con_comp_n[x,y] == -j ):
                counter = counter + 1
          number_pixel_j = np.count_nonzero(con_comp_n_in[:,:]==j) + np.count_nonzero(con_comp_n_in[:,:]==-j)
          if ( counter / number_pixel_j ) >= percentage:
            t_s_con_comp_in[ 0,con_comp_n_in==j ] = i
            t_s_con_comp_in[ 0,con_comp_n_in==-j ] = -i
            flag_comp_num_n[j-1] = 1
            flag_comp_num_n_1[i-1] = 1
            print("geometric center and overlapping match with percentage: number of pixel, index i and j, class", counter, i, j, class_n_1)
          if ( counter / number_pixel_j ) > max_union:
            max_union = ( counter / number_pixel_j )
            max_index = j        
    if max_union > 0:
      t_s_con_comp_in[ 0,con_comp_n_in==max_index ] = i
      t_s_con_comp_in[ 0,con_comp_n_in==-max_index ] = -i
      flag_comp_num_n[max_index-1] = 1
      flag_comp_num_n_1[i-1] = 1
      print("geometric center and overlapping match: number of pixel, index i and j, class", max_union, i, max_index, class_n_1) 
  
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in



def step2_2(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, int epsilon):
  
  cdef int x, y, class_n_1, class_n, counter, min_index
  cdef float mean_x_n, mean_y_n, mean_x_n_1, mean_y_n_1, mean_x_n_2, mean_y_n_2, min_distance, dist, dir_n_1_n_x, dir_n_1_n_y, dir_n_2_n_1_x, dir_n_2_n_1_y
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in
  
  if flag_comp_num_n_1[i-1] == 0:
    ind_n_1 = t_s_con_comp_in[1,:,:] == -i
    class_n_1 = int(seg_in[1,ind_n_1].min())
    mean_x_n_1 = 0
    mean_y_n_1 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[1,x,y] == i or t_s_con_comp[1,x,y] == -i:
          mean_x_n_1 += x
          mean_y_n_1 += y
          counter = counter + 1
    mean_x_n_1 /= counter
    mean_y_n_1 /= counter
    mean_x_n_2 = 0
    mean_y_n_2 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[2,x,y] == i or t_s_con_comp[2,x,y] == -i:
          mean_x_n_2 += x
          mean_y_n_2 += y
          counter = counter + 1
    mean_x_n_2 /= counter
    mean_y_n_2 /= counter
    min_distance = 2000
    min_index = 0
    dir_n_2_n_1_x =  mean_x_n_1 - mean_x_n_2
    dir_n_2_n_1_y =  mean_y_n_1 - mean_y_n_2   
    for j in range( 1, -int(con_comp_n_in.min())+1 ):   
      if flag_comp_num_n[j-1] == 0:
        ind_n = con_comp_n_in == -j
        class_n = int(seg_in[0,ind_n].min())
        if class_n_1 == class_n:
          mean_x_n = 0
          mean_y_n = 0
          counter = 0
          for x in range(imx):
            for y in range(imy):
              if con_comp_n[x,y] == j or con_comp_n[x,y] == -j:
                mean_x_n += x
                mean_y_n += y
                counter = counter + 1
          mean_x_n /= counter
          mean_y_n /= counter
          dir_n_1_n_x =  mean_x_n - mean_x_n_1
          dir_n_1_n_y =  mean_y_n - mean_y_n_1
          dist = ( dir_n_1_n_x**2 + dir_n_1_n_y**2 ) **0.5 + ( (dir_n_2_n_1_x - dir_n_1_n_x)**2 + (dir_n_2_n_1_y - dir_n_1_n_y)**2 )**0.5
          if dist < min_distance:
            min_distance = dist
            min_index = j    
    if min_index > 0 and min_distance < epsilon:
      t_s_con_comp_in[ 0,con_comp_n_in==min_index ] = i
      t_s_con_comp_in[ 0,con_comp_n_in==-min_index ] = -i
      flag_comp_num_n[min_index-1] = 1
      flag_comp_num_n_1[i-1] = 1
      print("geometric center match: distance with direction, index i and j, class", min_distance, i, min_index, class_n_1)   
  
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in
    


def simplified_step2_2(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, int epsilon):
  
  cdef int x, y, class_n_1, class_n, counter, min_index
  cdef float mean_x_n, mean_y_n, mean_x_n_1, mean_y_n_1, min_distance, dist
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in
  
  if flag_comp_num_n_1[i-1] == 0:
    ind_n_1 = t_s_con_comp_in[1,:,:] == -i
    class_n_1 = int(seg_in[1,ind_n_1].min())
    mean_x_n_1 = 0
    mean_y_n_1 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if t_s_con_comp[1,x,y] == i or t_s_con_comp[1,x,y] == -i:
          mean_x_n_1 += x
          mean_y_n_1 += y
          counter = counter + 1
    mean_x_n_1 /= counter
    mean_y_n_1 /= counter
    min_distance = 2000
    min_index = 0
    for j in range( 1, -int(con_comp_n_in.min())+1 ):
      if flag_comp_num_n[j-1] == 0:
        ind_n = con_comp_n_in == -j
        class_n = int(seg_in[0,ind_n].min())
        if class_n_1 == class_n:
          mean_x_n = 0
          mean_y_n = 0
          counter = 0
          for x in range(imx):
            for y in range(imy):
              if con_comp_n[x,y] == j or con_comp_n[x,y] == -j:
                mean_x_n += x
                mean_y_n += y
                counter = counter + 1
          mean_x_n /= counter
          mean_y_n /= counter
          dist = ( (mean_x_n - mean_x_n_1)**2 + (mean_y_n - mean_y_n_1)**2 )**0.5
          if dist < min_distance:
            min_distance = dist
            min_index = j
    if min_index > 0 and min_distance < epsilon:
      t_s_con_comp_in[ 0,con_comp_n_in==min_index ] = i
      t_s_con_comp_in[ 0,con_comp_n_in==-min_index ] = -i
      flag_comp_num_n[min_index-1] = 1
      flag_comp_num_n_1[i-1] = 1
      print("geometric center match: distance, index i and j, class", min_distance, i, min_index, class_n_1) 
      
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in
  
  
  
def step3(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, float percentage):
  
  cdef int x, y, class_n_1, class_n, counter, max_index
  cdef float number_pixel_j, max_union
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in

  if np.count_nonzero(t_s_con_comp_in[1,:,:]==-i) > 0:
    ind_n_1 = t_s_con_comp_in[1,:,:] == -i
    class_n_1 = int(seg_in[1,ind_n_1].min())
    max_union = 0
    max_index = 0
    for j in range( 1, -int(con_comp_n_in.min())+1 ):
      if flag_comp_num_n[j-1] == 0:
        ind_n = con_comp_n_in == -j
        class_n = int(seg_in[0,ind_n].min())
        if class_n_1 == class_n:
          counter = 0
          for x in range(imx):
            for y in range(imy):
              if ( t_s_con_comp[1,x,y] == i or t_s_con_comp[1,x,y] == -i ) and ( con_comp_n[x,y] == j or con_comp_n[x,y] == -j ):
                counter = counter + 1
          number_pixel_j = np.count_nonzero(con_comp_n_in[:,:]==j) + np.count_nonzero(con_comp_n_in[:,:]==-j)
          if ( counter / number_pixel_j ) >= percentage:
            t_s_con_comp_in[ 0,con_comp_n_in==j ] = i
            t_s_con_comp_in[ 0,con_comp_n_in==-j ] = -i
            flag_comp_num_n[j-1] = 1
            flag_comp_num_n_1[i-1] = 1
            print("overlapping match with percentage: number of pixel, index i and j, class", counter, i, j, class_n_1)
          if ( counter / number_pixel_j ) > max_union:
            max_union = ( counter / number_pixel_j )
            max_index = j
    if max_union > 0:
      t_s_con_comp_in[ 0,con_comp_n_in==max_index ] = i
      t_s_con_comp_in[ 0,con_comp_n_in==-max_index ] = -i
      flag_comp_num_n[max_index-1] = 1
      flag_comp_num_n_1[i-1] = 1
      print("overlapping match: number of pixel, index i and j, class", max_union, i, max_index, class_n_1)
      
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in
  
  
  
def step4_1(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, int eps_time, int reg_steps):
  
  cdef int x, y, c, class_n_1, class_n, counter, min_index
  cdef float mean_x_n, mean_y_n, min_distance, dist, a_x, a_y, b_x, b_y, pred_x, pred_y,
  cdef float[:,:] mean_field
  cdef float[:] mean_counter_field
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in
  
  if np.count_nonzero(t_s_con_comp_in[0,:,:]==-i) == 0:
    class_n_1 = -2
    for c in range(1,reg_steps+1):
      if np.count_nonzero(t_s_con_comp_in[c,:,:]==-i) != 0:
        ind_n_1 = t_s_con_comp_in[c,:,:] == -i
        class_n_1 = int(seg_in[c,ind_n_1].min())
        break
    if class_n_1 != -2:
      mean_field_in = np.zeros( (reg_steps,2), dtype="float32" )
      mean_field = mean_field_in
      mean_counter_field_in = np.zeros( (reg_steps), dtype="float32" )
      mean_counter_field = mean_counter_field_in
      for x in range(imx):
        for y in range(imy):
          for c in range(1,reg_steps+1):
            if t_s_con_comp[c,x,y] == i or t_s_con_comp[c,x,y] == -i:
              mean_field[c-1,0] += x
              mean_field[c-1,1] += y
              mean_counter_field[c-1] += 1
      mean_x_list = []
      mean_y_list = []
      mean_t_list = []
      for c in range(reg_steps):
        if mean_counter_field[c] > 0:
          mean_field[c,0] /= mean_counter_field[c]
          mean_field[c,1] /= mean_counter_field[c]
          mean_x_list.append(mean_field[c,0])
          mean_y_list.append(mean_field[c,1])
          mean_t_list.append(c)
      if len(mean_t_list) >= 2:
        b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
        b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
        pred_x = a_x + b_x * reg_steps
        pred_y = a_y + b_y * reg_steps
        min_distance = 2000
        min_index = 0
        for j in range( 1, -int(con_comp_n_in.min())+1 ):
          if flag_comp_num_n[j-1] == 0:
            ind_n = con_comp_n_in == -j
            class_n = int(seg_in[0,ind_n].min())
            if class_n_1 == class_n:
              mean_x_n = 0
              mean_y_n = 0
              counter = 0
              for x in range(imx):
                for y in range(imy):
                  if con_comp_n[x,y] == j or con_comp_n[x,y] == -j:
                    mean_x_n += x
                    mean_y_n += y
                    counter = counter + 1
              mean_x_n /= counter
              mean_y_n /= counter
              dist = ( (mean_x_n - pred_x)**2 + (mean_y_n - pred_y)**2 )**0.5
              if dist < min_distance:
                min_distance = dist
                min_index = j 
        if min_index > 0 and min_distance < eps_time:
          t_s_con_comp_in[ 0,con_comp_n_in==min_index ] = i
          t_s_con_comp_in[ 0,con_comp_n_in==-min_index ] = -i
          flag_comp_num_n[min_index-1] = 1
          flag_comp_num_n_1[i-1] = 1
          print("time series match: distance, index i and j, class", min_distance, i, min_index, class_n_1)    
  
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in  



def step4_2(int i, int imx, int imy, np.ndarray flag_comp_num_n_1_in, np.ndarray flag_comp_num_n_in, np.ndarray t_s_con_comp_in, np.ndarray seg_in, np.ndarray con_comp_n_in, float percentage, int reg_steps):
  
  cdef int x, y, c, class_n_1, class_n, counter, max_index, max_index_timestep, x_shift, y_shift, x_t, y_t
  cdef float a_x, a_y, b_x, b_y, pred_x, pred_y, number_pixel_j, max_union
  cdef float[:,:] mean_field
  cdef float[:] mean_counter_field
  cdef short int[:,:] shifted_comp
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  flag_comp_num_n = flag_comp_num_n_in 
  t_s_con_comp = t_s_con_comp_in
  con_comp_n = con_comp_n_in
  
  if np.count_nonzero(t_s_con_comp_in[0,:,:]==-i) == 0:
    class_n_1 = -2
    for c in range(1,reg_steps+1):
      if np.count_nonzero(t_s_con_comp_in[c,:,:]==-i) != 0:
        ind_n_1 = t_s_con_comp_in[c,:,:] == -i
        class_n_1 = int(seg_in[c,ind_n_1].min())
        break
    if class_n_1 != -2:
      mean_field_in = np.zeros( (reg_steps,2), dtype="float32" )
      mean_field = mean_field_in
      mean_counter_field_in = np.zeros( (reg_steps), dtype="float32" )
      mean_counter_field = mean_counter_field_in
      for x in range(imx):
        for y in range(imy):
          for c in range(1,reg_steps+1):
            if t_s_con_comp[c,x,y] == i or t_s_con_comp[c,x,y] == -i:
              mean_field[c-1,0] += x
              mean_field[c-1,1] += y
              mean_counter_field[c-1] += 1
      mean_x_list = []
      mean_y_list = []
      mean_t_list = []
      for c in range(reg_steps):
        if mean_counter_field[c] > 0:
          mean_field[c,0] /= mean_counter_field[c]
          mean_field[c,1] /= mean_counter_field[c]
          mean_x_list.append(mean_field[c,0])
          mean_y_list.append(mean_field[c,1])
          mean_t_list.append(c)
      if len(mean_t_list) >= 2:
        b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
        b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
        pred_x = a_x + b_x * reg_steps
        pred_y = a_y + b_y * reg_steps
        max_index_timestep = int(np.argmax(mean_counter_field))
        if np.count_nonzero(t_s_con_comp_in[0,:,:]==-i) == 0:
          x_shift = int(pred_x - mean_field[max_index_timestep,0])
          y_shift = int(pred_y - mean_field[max_index_timestep,1])
          shifted_comp_in = np.zeros((imx, imy), dtype="int16")
          shifted_comp = shifted_comp_in
          for x in range(imx):
            for y in range(imy):
              if t_s_con_comp[max_index_timestep+1,x,y] == i or t_s_con_comp[max_index_timestep+1,x,y] == -i:
                x_t = x + int(x_shift)
                y_t = y + int(y_shift)
                if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
                  shifted_comp[x_t,y_t] = i
          max_union = 0
          max_index = 0
          for j in range( 1, -int(con_comp_n_in.min())+1 ):
            if flag_comp_num_n[j-1] == 0:
              ind_n = con_comp_n_in == -j
              class_n = int(seg_in[0,ind_n].min())
              if class_n_1 == class_n:
                counter = 0
                for x in range(imx):
                  for y in range(imy):
                    if ( shifted_comp[x,y] == i ) and ( con_comp_n[x,y] == j or con_comp_n[x,y] == -j ):
                      counter = counter + 1
                number_pixel_j = np.count_nonzero(con_comp_n_in[:,:]==j) + np.count_nonzero(con_comp_n_in[:,:]==-j)
                if ( counter / number_pixel_j ) >= percentage:
                  t_s_con_comp_in[ 0,con_comp_n_in==j ] = i
                  t_s_con_comp_in[ 0,con_comp_n_in==-j ] = -i
                  flag_comp_num_n[j-1] = 1
                  flag_comp_num_n_1[i-1] = 1
                  print("time series and overlapping match with percentage: number of pixel, index i and j, class", counter, i, j, class_n_1)
                if ( counter / number_pixel_j ) > max_union:
                  max_union = ( counter / number_pixel_j )
                  max_index = j 
          if max_union > 0:
            t_s_con_comp_in[ 0,con_comp_n_in==max_index ] = i
            t_s_con_comp_in[ 0,con_comp_n_in==-max_index ] = -i
            flag_comp_num_n[max_index-1] = 1
            flag_comp_num_n_1[i-1] = 1
            print("time series and overlapping match: number of pixel, index i and j, class", max_union, i, max_index, class_n_1)    
  
  return t_s_con_comp_in, flag_comp_num_n_1_in, flag_comp_num_n_in  
  
      
      
def compute_t_s_metrics( t_s_con_comp_in, probs_in, gt, max_con_comp ):    
      
  cdef int imx, imy, imc, counter, x, y, i, ic, n_in, n_bd, I, U, c, class_j, class_k
  cdef short int[:,:] t_s_con_comp
  cdef float [:,:,:] probs
  cdef long int[:,:] gt_con_comp
  imx = probs_in.shape[0]
  imy = probs_in.shape[1]
  imc = probs_in.shape[2]
  t_s_con_comp_in = np.asarray( t_s_con_comp_in, dtype="int16" )
  t_s_con_comp = t_s_con_comp_in
  probs_in = np.asarray( probs_in, dtype="float32" )
  probs  = probs_in
  
  gt = np.asarray( gt, dtype="uint8" )
  if np.sum(gt) == 0:
    gt = []
  
  # seg has 255 on all pixels where gt has 255
  seg = np.asarray( prediction(probs_in, gt, ignore=True ), dtype="uint8" )
  
  heatmaps = { "E": entropy( probs_in ), "M": probdist( probs_in ), "V": variation_ratio( probs_in ) }  
  
  t_s_metrics = { "iou": list([]), "iou0": list([]), "class": list([]), "mean_x": list([]), "mean_y": list([]) } 
  
  for m in list(heatmaps)+["S"]:
    t_s_metrics[m          ] = list([])
    t_s_metrics[m+"_in"    ] = list([])
    t_s_metrics[m+"_bd"    ] = list([])
    t_s_metrics[m+"_rel"   ] = list([])
    t_s_metrics[m+"_rel_in"] = list([])
    
  for i in range(imc):
    t_s_metrics['cprob'+str(i)] = list([])
  
  if gt != []:
    gt_con_comp_in = ms.label(gt, background=-2)
    gt_con_comp = gt_con_comp_in
    
  for i in range( 1, max_con_comp+1 ):
    
    for m in t_s_metrics:
      t_s_metrics[m].append( 0 )
      
    if ( np.count_nonzero(t_s_con_comp_in[:,:]==i) + np.count_nonzero(t_s_con_comp_in[:,:]==-i) ) > 0:
      
      n_in = 0
      n_bd = 0
      I = 0
      U = 0
      gt_U_list = []
      comp_U_list = []
      
      ind_bd = t_s_con_comp_in[:,:] == -i
      c = int(seg[ind_bd].min())
      
      for x in range(imx):
        for y in range(imy):
          if ( t_s_con_comp[x,y] == i or t_s_con_comp[x,y] == -i ) and seg[x,y] != 255:
            if t_s_con_comp[x,y] == i:
              for h in heatmaps:
                t_s_metrics[h+"_in"][-1] += heatmaps[h][x,y]
              n_in += 1
            elif t_s_con_comp[x,y] == -i:
              for h in heatmaps:
                t_s_metrics[h+"_bd"][-1] += heatmaps[h][x,y]
              n_bd += 1
            for ic in range(imc):
              t_s_metrics["cprob"+str(ic)][-1] += probs[x,y,ic]
            t_s_metrics["mean_x"][-1] += x
            t_s_metrics["mean_y"][-1] += y
            if gt != []:
              if gt[x,y] == c:
                I += 1
                if gt_con_comp[x,y] not in gt_U_list:
                  gt_U_list.append(gt_con_comp[x,y])   
              U += 1
      
      if gt != []:
        for j in gt_U_list:
          ind_j = gt_con_comp_in[:,:] == j
          class_j = int(gt[ind_j].min())
          
          for k in range(1, -int(t_s_con_comp_in.min())+1):
            
            if ( np.count_nonzero(t_s_con_comp_in[:,:]==k) + np.count_nonzero(t_s_con_comp_in[:,:]==-k) ) > 0:
              ind_k_in = t_s_con_comp_in[:,:] == k
              ind_k_bd = t_s_con_comp_in[:,:] == -k
              ind_k = ind_k_in + ind_k_bd
              class_k = int(seg[ind_k_bd].min())
              if k != i and class_j == class_k and np.count_nonzero( np.logical_and(ind_j,ind_k) ) > 0:
                comp_U_list.append(k)

        for x in range(imx):
          for y in range(imy):
            if (t_s_con_comp[x,y] != i) and (t_s_con_comp[x,y] != -i) and (gt_con_comp[x,y] in gt_U_list) and (t_s_con_comp[x,y] not in comp_U_list) and seg[x,y] != 255:
              U += 1
         
      t_s_metrics["class"   ][-1] = c
      if gt != []:
        if U > 0:
          t_s_metrics["iou"     ][-1] = float(I) / float(U)
          t_s_metrics["iou0"    ][-1] = int(I == 0)
        else:
          t_s_metrics["iou"     ][-1] = 0
          t_s_metrics["iou0"    ][-1] = int(I == 0)
      else:
        t_s_metrics["iou"     ][-1] = -1
        t_s_metrics["iou0"    ][-1] = -1
        
      if n_bd > 0:
        t_s_metrics["S"       ][-1] = n_in + n_bd
        t_s_metrics["S_in"    ][-1] = n_in
        t_s_metrics["S_bd"    ][-1] = n_bd
        t_s_metrics["S_rel"   ][-1] = float( n_in + n_bd ) / float(n_bd)
        t_s_metrics["S_rel_in"][-1] = float( n_in ) / float(n_bd)
        t_s_metrics["mean_x"][-1] /= ( n_in + n_bd )
        t_s_metrics["mean_y"][-1] /= ( n_in + n_bd )
        
        for nc in range(imc):
          t_s_metrics["cprob"+str(nc)][-1] /= ( n_in + n_bd )
        
        for h in heatmaps:
          t_s_metrics[h][-1] = (t_s_metrics[h+"_in"][-1] + t_s_metrics[h+"_bd"][-1]) / float( n_in + n_bd )
          if ( n_in > 0 ):
            t_s_metrics[h+"_in"][-1] /= float(n_in)
          t_s_metrics[h+"_bd"][-1] /= float(n_bd)
            
          t_s_metrics[h+"_rel"   ][-1] = t_s_metrics[h      ][-1] * t_s_metrics["S_rel"   ][-1]
          t_s_metrics[h+"_rel_in"][-1] = t_s_metrics[h+"_in"][-1] * t_s_metrics["S_rel_in"][-1]
    
    else:
      if gt == []:
        t_s_metrics["iou"     ][-1] = -1
        t_s_metrics["iou0"    ][-1] = -1
      
      
  return t_s_metrics
        


def entropy( probs ):
  E = np.sum( np.multiply( probs, np.log(probs+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/probs.shape[-1])
  return np.asarray( E, dtype="float32" )



def probdist( probs ):
  cdef int i, j
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] + P[ i, j, A[i,j,-2] ] )
  return arrayD



def variation_ratio( probs ):
  cdef int i, j
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] )
  return arrayD
