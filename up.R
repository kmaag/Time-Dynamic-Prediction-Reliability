args = commandArgs(trailingOnly=TRUE)

library(MBA)
library(gstat)
library(sp)
library(automap)
library(randomForest) 
library(UBL)
library(RcppCNPy)
library(lattice)
library(grid)
library(DMwR)

Xa_path <- paste(args[2], "Xa_train_run",args[1],".npy",sep = "", collapse = NULL)
ya_path <- paste(args[2], "ya_train_run",args[1],".npy",sep = "", collapse = NULL)
Xa_in <- npyLoad(Xa_path)
ya_in <- npyLoad(ya_path)
Xa_ya_tmp <- cbind ( ya_in , Xa_in )
Xa_ya_as_frame <- as.data.frame(Xa_ya_tmp)

matrix_rel <- matrix(0, ncol = 2, nrow = 0)
matrix_rel <- rbind(matrix_rel, c(0.1, 0.5))
matrix_rel <- rbind(matrix_rel, c(0.6, 0.2))
matrix_rel <- rbind(matrix_rel, c(1, 0.2))

Xa_ya_as_frame_out <- SmoteRegress(ya_in~., Xa_ya_as_frame, dist="Euclidean",rel=matrix_rel,C.perc=list(0.5,2.5),k=5)

for(i in 1:8){
  if(i==7 || i==8){
    #print(4)
    matrix_rel <- matrix(0, ncol = 2, nrow = 0)
    matrix_rel <- rbind(matrix_rel, c(0.1, 0.2))
    matrix_rel <- rbind(matrix_rel, c(0.6, 0.5))
    matrix_rel <- rbind(matrix_rel, c(1, 0.8)) }  
  else if(i==5 || i==6){
    #print(3)
    matrix_rel <- matrix(0, ncol = 2, nrow = 0)
    matrix_rel <- rbind(matrix_rel, c(0.1, 0.1))
    matrix_rel <- rbind(matrix_rel, c(0.6, 0.7))
    matrix_rel <- rbind(matrix_rel, c(1, 0.9)) }
  else if(i==3 || i==4){
    #print(2)
    matrix_rel <- matrix(0, ncol = 2, nrow = 0)
    matrix_rel <- rbind(matrix_rel, c(0.1, 0.9))
    matrix_rel <- rbind(matrix_rel, c(0.6, 0.5))
    matrix_rel <- rbind(matrix_rel, c(1, 0.3)) }
  else if(i==1 || i==2){
    #print(1)
    matrix_rel <- matrix(0, ncol = 2, nrow = 0)
    matrix_rel <- rbind(matrix_rel, c(0.1, 0.7))
    matrix_rel <- rbind(matrix_rel, c(0.6, 0.5))
    matrix_rel <- rbind(matrix_rel, c(1, 0.1)) }
    
  out_Xa_ya_as_frame_tmp <- SmoteRegress(ya_in~., Xa_ya_as_frame, dist="Euclidean",rel=matrix_rel,C.perc=list(0.5,2.5),k=5)
  Xa_ya_as_frame_out <- rbind ( Xa_ya_as_frame_out , out_Xa_ya_as_frame_tmp ) }

ya_out = Xa_ya_as_frame_out[,1]
num_metrics <- dim(Xa_ya_as_frame_out)[2]
Xa_new_as_frame = Xa_ya_as_frame_out[,c(2:num_metrics)]

save_Xa <- paste(args[2], "Xa_A_run",args[1],".npy",sep = "", collapse = NULL)
save_ya <- paste(args[2], "ya_A_run",args[1],".npy",sep = "", collapse = NULL)
Xa_out <- data.matrix(Xa_new_as_frame)
npySave(save_Xa, Xa_out)
npySave(save_ya, ya_out) 
