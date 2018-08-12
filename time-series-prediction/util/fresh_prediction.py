#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM




def fit_lstm(X_train, Y_train, X_val, Y_val, neurons, param, verbose=False):

    X_train = X_train.reshape(X_train.shape[0], param.n_seq_steps, param.n_features) # One step and n_lag features
    X_val = X_val.reshape(X_val.shape[0], param.n_seq_steps, param.n_features) # One step and n_lag features
    
    # design network
    history_list = list()
    
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(param.batch_size, param.n_seq_steps, param.n_features), 
                   stateful=param.b_stateful,
                   recurrent_dropout= param.ratio_rec_dropout,
                   dropout = param.ratio_input_dropout)) # X.shape[1] as step, X.shape[2] as feature
    model.add(Dense(param.n_pre_steps, activation='linear')) # pre_steps
    model.compile(loss='mean_squared_error', optimizer='adam')
        
    if verbose:
        print("Shape of X_train in fit_lstm: ", X_train.shape)
        print("Shape of Y_train in fit_lstm: ", Y_train.shape)     
        print("Shape of X_val in fit_lstm: ", X_val.shape)
        print("Shape of Y_val in fit_lstm: ", Y_val.shape)             
        model.summary()
        
    # fit network
    for i in range(param.n_epochs):
        if i % 10 == 0:
            print("# of epochs: ", i)
        history = model.fit(X_train, Y_train, epochs=1, batch_size= param.batch_size, 
                            validation_data=(X_val, Y_val), 
                            verbose=0, shuffle= param.b_shuffle) # stateful
        history_list.append(history)
        model.reset_states()
        
    return model, history_list

# stacked lstm model
def fit_lstm2(X_train, Y_train, X_val, Y_val, neurons, param, verbose=False):

    X_train = X_train.reshape(X_train.shape[0], param.n_seq_steps, param.n_features) # One step and n_lag features
    X_val = X_val.reshape(X_val.shape[0], param.n_seq_steps, param.n_features) # One step and n_lag features
    
    # design network
    history_list = list()
    
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(param.batch_size, param.n_seq_steps, param.n_features), 
                   stateful=param.b_stateful,
                   recurrent_dropout= param.ratio_rec_dropout,
                   dropout = param.ratio_input_dropout,
                   return_sequences=True)) 
    model.add(LSTM(neurons, 
                   stateful=param.b_stateful,
                   recurrent_dropout= param.ratio_rec_dropout))                    
    model.add(Dense(param.n_pre_steps, activation='linear')) # pre_steps
    model.compile(loss='mean_squared_error', optimizer='adam')
        
    if verbose:
        print("Shape of X_train in fit_lstm: ", X_train.shape)
        print("Shape of Y_train in fit_lstm: ", Y_train.shape)     
        print("Shape of X_val in fit_lstm: ", X_val.shape)
        print("Shape of Y_val in fit_lstm: ", Y_val.shape)             
        model.summary()
        
    # fit network
    for i in range(param.n_epochs):
        if i % 10 == 0:
            print("# of epochs: ", i)
        history = model.fit(X_train, Y_train, epochs=1, batch_size= param.batch_size, 
                            validation_data=(X_val, Y_val), 
                            verbose=0, shuffle= param.b_shuffle) # stateful
        history_list.append(history)
        model.reset_states()
        
    return model, history_list

 
# Make a forcast model to handle a batch size of 1
def get_forcast_lstm(old_weights, param ):
    # re-define the batch size
    n_forcast_batch = 1
    # re-define model
    new_model = Sequential()
    new_model.add(LSTM(param.n_neurons, batch_input_shape=(n_forcast_batch, param.n_seq_steps, param.n_features), stateful=param.b_stateful))
    new_model.add(Dense(param.n_pre_steps, activation='linear'))
    # copy weights

    new_model.set_weights(old_weights)
    # compile model
    new_model.compile(loss='mean_squared_error', optimizer='adam')
    return new_model   

# Retrieve stacked lstm model
def get_forcast_lstm2(old_weights, param ):
    # re-define the batch size
    n_forcast_batch = 1
    # re-define model
    new_model = Sequential()
    new_model.add(LSTM(param.n_neurons, batch_input_shape=(n_forcast_batch, param.n_seq_steps, param.n_features), 
                       stateful=param.b_stateful,
                       return_sequences=True))
    new_model.add(LSTM(param.n_neurons, 
                       stateful=param.b_stateful,
                       ))                       
    new_model.add(Dense(param.n_pre_steps, activation='linear'))
    # copy weights

    new_model.set_weights(old_weights)
    # compile model
    new_model.compile(loss='mean_squared_error', optimizer='adam')
    return new_model


# make one forecast with an LSTM
def forecast_lstm(model, X, param, verbose=False):
    if verbose:
        print("X shape in forcast: " , X.shape)
    
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(param.forcast_batch_size, param.n_seq_steps, param.n_features)
    # make forecast
    forecast = model.predict(X, batch_size= param.forcast_batch_size)
    # convert to array
    
    if verbose:
        print("forecast shape in forcast : " , forecast.shape)

    return forecast



def make_forecasts(model, test_X, param, verbose=False):
    forecasts = list()
    for i in range(0, len(test_X), param.forcast_batch_size): 
        X = test_X[i:i + param.forcast_batch_size]
        if verbose:
            print("shape of X: \n", X.shape)
        
        forecast = forecast_lstm(model, X, param, verbose)
        
        if verbose:
            print("shape of forecast: \n", forecast.shape)
            
            
        forecasts.append(forecast)
    return forecasts
 

from sklearn.metrics import mean_squared_error
def evaluate_mse(true_value, predict, verbose=False):
    mse = mean_squared_error(true_value, predict)
    return mse

import numpy as np
def invert_scale(scaler, yhat, verbose=False):
    output_item = np.array(yhat).reshape(1,1)
    invert_output = scaler.inverse_transform(output_item)
    invert_output = np.squeeze(invert_output)
    return invert_output



def convert_prediction(scaler, output, param, verbose=True):
    prediction_list = list()
    prediction_step_list = list()
    
    for i in range(param.n_pre_steps):
        for j in range(len(output)):
            output_item = output[j,i]
            invert_output = invert_scale(scaler, output_item, verbose=False)
            prediction_step_list.append(invert_output)
        prediction_list.append(prediction_step_list)
        prediction_step_list = list()
        
    prediction_list = np.array(prediction_list)
    prediction_list = prediction_list.reshape(prediction_list.shape[1], prediction_list.shape[0])
    
    return prediction_list   
 

def make_mul_index(mul, lens):
    idx_list = list()
    serial = 0 ; idx =0
    while idx < lens:
        idx_list.append(idx)
        serial += 1
        idx = serial * mul
        
    return idx_list
   
   
import math
def predict(old_weights, param, scaler, x_data_scale, y_data_true, verbose=False ):
    
    #forcast_model = get_forcast_lstm(old_weights, param)
    
    forcast_model = get_forcast_lstm2(old_weights, param) # Retrieve stacked lstm model
    
    forecasts_scale = make_forecasts(forcast_model,  x_data_scale, param, verbose=False)
    # convert to np array
    forecasts_scale_np = np.array(forecasts_scale)
    # convert to 2d from 3d
    forcast_row_shape = forecasts_scale_np.shape[0] * forecasts_scale_np.shape[1]
    forcast_col_shape = forecasts_scale_np.shape[2]
    forecasts_scale_2d_np = forecasts_scale_np.reshape(forcast_row_shape, forcast_col_shape)

    predictions = convert_prediction(scaler, forecasts_scale_2d_np, param, verbose=True)
    test_idx_list = make_mul_index(mul= param.n_pre_steps, lens = len(predictions))

    unique_predict = predictions[test_idx_list]
    unique_predict_vector = unique_predict.reshape(-1,1)

    if verbose:
        print("Shape of forecasts_scale: ",len(forecasts_scale))            
        print("Shape of forecasts_scale_np: ", forecasts_scale_np.shape)            
        print("Shape of forecasts_scale_2d_np: ", forecasts_scale_2d_np.shape)
        print("Shape of predictions: \n", predictions.shape)
        print("Shape of unique_predict: \n", unique_predict.shape)        
        print("Shape of unique_predict_vector: \n", unique_predict_vector.shape)        
        print("Shape of y_data_true: ", y_data_true.shape)        
        #print("forecasts_scale_2d_np: ", forecasts_scale_2d_np[0:8])            
        #print("predictions: \n",predictions[0:8])

    mse = evaluate_mse(y_data_true, unique_predict_vector, verbose=False)
    rmse = math.sqrt(mse)
    print('TEST RMSE: %.3f' % (rmse))
    
    return rmse, unique_predict_vector
 
 
# Evaluate given a data set
def evaluate(title, old_weights, param, scaler, x_data_scale, y_data_raw,  verbose=False):
        
    rmse, prediction = predict(old_weights, param,scaler, x_data_scale, y_data_raw,  verbose= verbose )    

    display_obs_pred(title, rmse, param, y_data_raw, prediction, x_data_scale)

# Display observation and prediction        
import matplotlib.pyplot as plt
import seaborn;seaborn.set()
plt.rcParams["figure.figsize"] = [12, 4]

def display_obs_pred(title, best_error_score, param,  y_test_true, best_predictions, x_train_scale,TEST_SIZE=0):
    plt.title(title + 
              " Prediction Quality: {:.2f} RMSE with {} lags, {} pre-steps, {} seq-steps, {} features, {} batch_size \n \
              State: {}, Shuffle: {} \
              Train size: {}, \
              Test size:{} hours".
             format(best_error_score, param.n_lags, param.n_pre_steps, param.n_seq_steps,param.n_features, param.batch_size, 
                    param.b_stateful, param.b_shuffle,
                    len(x_train_scale), TEST_SIZE))
    plt.plot(y_test_true, label = 'Observed', color='#006699')
    plt.plot(best_predictions, label='Prediction', color='#ff0066')
    plt.legend(loc='best')
    plt.show()
   