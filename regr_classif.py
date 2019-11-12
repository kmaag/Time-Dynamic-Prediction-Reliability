import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPClassifier
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense

from defs_global import SETUP



def classification( X_train, y_train, X_val, y_val, X_test_R, X_test_P ):
  
  if SETUP.CLASSIFICATION_TYPE == "NN_L2":
    model = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', solver='adam', alpha=0.4, batch_size=128, max_iter=200)
    model.fit( X_train, y_train )
    
  elif SETUP.CLASSIFICATION_TYPE == "GB":
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
    model.fit( X_train, y_train )
  
  elif SETUP.CLASSIFICATION_TYPE == "LR_L1":
    lambdas = get_lambdas( 50, -4.2, 0.8 )
    max_acc = -1
    best_model = []
    for i in range(len(lambdas)):
      model = linear_model.LogisticRegression(C=lambdas[i], penalty='l1', solver='saga', max_iter=1000, tol=1e-3 )
      model.fit( X_train, y_train )
      y_val_pred = model.predict_proba(X_val)
      tmp = np.mean( np.argmax(y_val_pred,axis=-1)==y_val )
      if tmp > max_acc:
        max_acc = tmp
        best_model = model
    model = best_model
    
  y_train_pred = model.predict_proba(X_train)
  y_val_pred = model.predict_proba(X_val)
  y_test_R_pred = model.predict_proba(X_test_R)
  y_test_P_pred = model.predict_proba(X_test_P)
  return y_train_pred, y_val_pred, y_test_R_pred, y_test_P_pred



def regression( X_train, y_train, X_val, y_val, X_test_R, X_test_P, num_prev_frames = SETUP.NUMBER_OF_FRAMES ):
  
  if SETUP.REGRESSION_TYPE == "NN_L2":
    num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
    X_train = X_train.reshape(X_train.shape[0], num_prev_frames+1, num_metrics )
    X_val = X_val.reshape(X_val.shape[0], num_prev_frames+1, num_metrics )
    X_test_R = X_test_R.reshape(X_test_R.shape[0], num_prev_frames+1, num_metrics )
    X_test_P = X_test_P.reshape(X_test_P.shape[0], num_prev_frames+1, num_metrics )
    input_shape  = (X_train.shape[1], X_train.shape[2])
    inp = Input(input_shape)
    wdecay=1e-3
    dropout=0.25
    y = inp
    y = Conv1D(filters=16, kernel_size=(5,), padding='same', strides=1, kernel_regularizer=regularizers.l2(wdecay), activation='relu')(inp)
    y = Flatten()(y)
    y = Dense( 50, kernel_regularizer=regularizers.l2(wdecay), activation='relu' )(y)
    y = Dense( 1 )(y)
    model = Model(inputs=inp,outputs=y)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    model.fit(X_train, y_train, epochs=200, validation_data=(X_val,y_val), batch_size=128) 
    #print(model.summary())
    
  elif SETUP.REGRESSION_TYPE == "NN_L1":
    num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
    X_train = X_train.reshape(X_train.shape[0], num_prev_frames+1, num_metrics )
    X_val = X_val.reshape(X_val.shape[0], num_prev_frames+1, num_metrics )
    X_test_R = X_test_R.reshape(X_test_R.shape[0], num_prev_frames+1, num_metrics )
    X_test_P = X_test_P.reshape(X_test_P.shape[0], num_prev_frames+1, num_metrics )
    input_shape  = (X_train.shape[1], X_train.shape[2])
    inp = Input(input_shape)
    weight=1e-4
    dropout=0.25
    y = inp
    y = Conv1D(filters=16, kernel_size=(5,), padding='same', strides=1, kernel_regularizer=regularizers.l1(weight), activation='relu')(inp)
    y = Flatten()(y)
    y = Dense( 50, kernel_regularizer=regularizers.l1(weight), activation='relu' )(y)
    y = Dense( 1 )(y)
    model = Model(inputs=inp,outputs=y)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    model.fit(X_train, y_train, epochs=200, validation_data=(X_val,y_val), batch_size=128) 
    
  elif SETUP.REGRESSION_TYPE == "GB":
    model = xgb.XGBRegressor(max_depth=5, colsample_bytree=0.5, n_estimators=100, reg_alpha=0.4, reg_lambda=0.4)
    model.fit(X_train, y_train)
    
  elif SETUP.REGRESSION_TYPE == "LR_L2":
    lambdas = get_lambdas( 50, -4.2, 0.8 )
    min_mse = 1000
    best_model = []
    for i in range(len(lambdas)):
      model = linear_model.Ridge(alpha=lambdas[i], max_iter=1000, tol=1e-3)
      model.fit(X_train,y_train)
      y_val_pred = np.clip( model.predict(X_val), 0, 1 )
      tmp = 1.0/len(y_val) * np.sum( (y_val-y_val_pred)**2 )
      if tmp < min_mse:
        min_mse = tmp
        best_model = model
    model = best_model
    
  elif SETUP.REGRESSION_TYPE == "LR_L1":
    lambdas = get_lambdas( 50, -4.2, 0.8 )
    min_mse = 1000
    best_model = []
    for i in range(len(lambdas)):
      model = linear_model.Lasso(alpha=lambdas[i], max_iter=1e5, tol=1e-3)
      model.fit(X_train,y_train)
      y_val_pred = np.clip( model.predict(X_val), 0, 1 )
      tmp = 1.0/len(y_val) * np.sum( (y_val-y_val_pred)**2 )
      if tmp < min_mse:
        min_mse = tmp
        best_model = model
    model = best_model
  
  elif SETUP.REGRESSION_TYPE == "LR":
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)
    
  y_train_pred = np.clip( model.predict(X_train), 0, 1 )
  y_val_pred = np.clip( model.predict(X_val), 0, 1 )
  y_test_R_pred = np.clip( model.predict(X_test_R), 0, 1 )
  y_test_P_pred = np.clip( model.predict(X_test_P), 0, 1 )
  return y_train_pred, y_val_pred, y_test_R_pred, y_test_P_pred, model

    

def get_lambdas( n_steps, min_pow, max_pow ):
  m = interp1d([0,n_steps-1],[min_pow,max_pow])
  lambdas = [10 ** m(i).item() for i in range(n_steps)]
  return lambdas
