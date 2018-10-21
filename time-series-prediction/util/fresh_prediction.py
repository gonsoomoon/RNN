#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM


def fit_lstm(X_train, Y_train, X_val, Y_val, neurons, param, verbose=False):
    """
    A lstm followed by a dense layer with dropout parameters
    training by an epoch, resetting states
    """
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
    """
    Two lstm layers
    """

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

# stacked MLP model


def fit_mlp(X_train, Y_train, X_val, Y_val, neurons, param, verbose=False):
    """
    Three hidden layers and one dropout applied to the first hidden layer
    """
    mdl = Sequential()
    mdl.add(Dense(neurons, input_dim=param.n_lags, activation='tanh'))
    mdl.Dropout(param.ratio_dropout)
    mdl.add(Dense(neurons, activation='tanh'))
    mdl.add(Dense(param.pre_step))
    mdl.compile(loss='mean_squared_error', optimizer='adam')
    mdl.fit(X_train, Y_train, epochs=nb_epochs, batch_size=nb_batch_size, verbose=verbose, shuffle=False)
    return mdl
 
def fit_mlp2(X_train, Y_train, X_val, Y_val, neurons, param, verbose=False):
    """
    Two lstm layers followed by one dense layer
    """

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
 

def get_forcast_lstm(old_weights, param ):
    """
    Make a forcast model to handle a batch size of 1
    """
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

def get_forcast_lstm2(old_weights, param ):
    """
    Retrieve the stacked lstm model
    """
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
    """
    A main function for a forcast
    """
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
    """
    evaluate mse
    """
    mse = mean_squared_error(true_value, predict)
    return mse

import numpy as np
def invert_scale(scaler, yhat, verbose=False):
    """
    Invert only one feature
    """
    output_item = np.array(yhat).reshape(1,1)
    invert_output = scaler.inverse_transform(output_item)
    invert_output = np.squeeze(invert_output)
    return invert_output



def convert_prediction(scaler, output, param, verbose=True):
    """
    Invert to an unscaled prediction
    """
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
 
from math import sqrt
from numpy import concatenate

def get_inv_y(y, x_features, scaler):
    """
    Compute an inverted y with a scaled y and X features using a scaler
    """
    y_x_features = concatenate((y, x_features), axis=1)
    inv_y_x_features = scaler.inverse_transform(y_x_features)
    inv_y = inv_y_x_features[:,0]
    return inv_y
def compute_rmse(yhat, test_y, x_features, scaler):
    """
    Compute rmse
    """
    #print("yhat: {}, x_features: {}".format(yhat.shape, x_features.shape))
    inv_yhat = get_inv_y(yhat, x_features, scaler)
    inv_y = get_inv_y(test_y, x_features, scaler)
    #print("inv_y: {}".format(inv_y))
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)## Test RMSE: 26.303    
    return rmse

def compute_all_rmse(yhat, test_y, test_X, scaler, param):
    """
    This function is not used but it is defined to compute all rmse for future use
    """
    
    list_rmse = list()
    test_X1 = test_X.reshape((test_X.shape[0], param.n_lags * param.n_features))
    x_features = test_X1[:,1:2]
    print("x_features shape: ", x_features.shape)
        
    for i in range(yhat.shape[1]): # number of pre-steps
        print(i)
        yhat_item = yhat[:,i]
        yhat_item = yhat_item.reshape((len(yhat_item), 1))
        test_y_item = test_y[:,i]
        test_y_item = test_y_item.reshape((len(test_y_item), 1))        
        rmse = compute_rmse(yhat_item, test_y_item, x_features, scaler)
        list_rmse.append(rmse)

    return list_rmse   
   
def invert_y(yhat, test_X, scaler, param, verbose=False):
    """
    A main function to get an unscaled y
    """
    prediction_list = list()
    test_X1 = test_X.reshape((test_X.shape[0], param.n_lags * param.n_features))
    x_features = test_X1[:,1:2]
    #print("x_features shape: ", x_features.shape)
        
    for i in range(yhat.shape[1]): # number of pre-steps
        yhat_item = yhat[:,i]
        yhat_item = yhat_item.reshape((len(yhat_item), 1))
        inv_yhat = get_inv_y(yhat_item, x_features, scaler)
        prediction_list.append(inv_yhat) 
        
    #print(prediction_list)
        
    return prediction_list

import math
def predict2(old_weights, param, scaler, x_data_scale, y_data_scale, verbose=False ):
    """
    multiple feature function
    """
    forcast_model = get_forcast_lstm2(old_weights, param) # Retrieve stacked lstm model
    
    forecasts_scale = make_forecasts(forcast_model,  x_data_scale, param, verbose=False)
    # convert to np array
    forecasts_scale_np = np.array(forecasts_scale)
    # convert to 2d from 3d
    forcast_row_shape = forecasts_scale_np.shape[0] * forecasts_scale_np.shape[1]
    forcast_col_shape = forecasts_scale_np.shape[2]
    forecasts_scale_2d_np = forecasts_scale_np.reshape(forcast_row_shape, forcast_col_shape)

    # invert to an original
    predictions = invert_y(forecasts_scale_2d_np, x_data_scale, scaler,  param, verbose=True)
    predictions = np.array(predictions).T
    inv_y = invert_y(y_data_scale, x_data_scale, scaler,  param, verbose=True)
    inv_y = np.array(inv_y).T
    
    #print(predictions)
    print("type predictions: {}".format(type(predictions)))
    print("Shape of predictions: \n", predictions.shape)
    

    
    test_idx_list = make_mul_index(mul= param.n_pre_steps, lens = len(predictions))

    unique_predict = predictions[test_idx_list]
    unique_predict_vector = unique_predict.reshape(-1,1)
    unique_inv_y = inv_y[test_idx_list]
    unique_inv_y_vector = unique_inv_y.reshape(-1,1)

    if verbose:
        print("Shape of forecasts_scale: ",len(forecasts_scale))            
        print("Shape of forecasts_scale_np: ", forecasts_scale_np.shape)            
        print("Shape of forecasts_scale_2d_np: ", forecasts_scale_2d_np.shape)
        print("Shape of predictions: \n", predictions.shape)
        print("Shape of unique_predict: \n", unique_predict.shape)        
        print("Shape of unique_predict_vector: \n", unique_predict_vector.shape)        
        print("Shape of inv_y: ", inv_y.shape)        
        print("forecasts_scale_2d_np: ", forecasts_scale_2d_np[0:8])            
        print("predictions: \n",predictions[0:8])


    mse = evaluate_mse(unique_inv_y_vector, unique_predict_vector, verbose=False)
    rmse = math.sqrt(mse)
    print('TEST RMSE: %.3f' % (rmse))
    
    
    
    return rmse, unique_predict_vector, unique_inv_y_vector


def predict(old_weights, param, scaler, x_data_scale, y_data_true, verbose=False ):
    """
    predict a model on which one feature is used for training
    """
    
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
 
def make_mul_index(mul, lens):
    """
    Get a multiple of a number
    """
    idx_list = list()
    serial = 0 ; idx =0
    while idx < lens:
        idx_list.append(idx)
        serial += 1
        idx = serial * mul
        
    return idx_list   
 
 
# Evaluate given a data set
def evaluate(title, old_weights, param, scaler, x_data_scale, y_data_raw,  verbose=False):
    """
    predict and show a chart with an observation
    """
        
    rmse, prediction = predict(old_weights, param,scaler, x_data_scale, y_data_raw,  verbose= verbose )    

    display_obs_pred(title, rmse, param, y_data_raw, prediction, x_data_scale)
    
def evaluate_with_model2(title, model, param, scaler, x_data_scale, y_data_scale,  verbose=False):
    model_weights = model.get_weights()
        
    rmse, inv_y_predict, inv_y_true = predict2(model_weights, param,scaler, x_data_scale, y_data_scale,  verbose= verbose )    

    display_obs_pred(title, rmse, param, inv_y_predict, inv_y_true, x_data_scale)

def evaluate_with_model(title, model, param, scaler, x_data_scale, y_data_raw,  verbose=False):
    """
    With a model passed, evaluate it
    """
    model_weights = model.get_weights()
        
    rmse, prediction = predict(model_weights, param,scaler, x_data_scale, y_data_raw,  verbose= verbose )    

    display_obs_pred(title, rmse, param, y_data_raw, prediction, x_data_scale)    


# Display observation and prediction        
import matplotlib.pyplot as plt
import seaborn;seaborn.set()
plt.rcParams["figure.figsize"] = [12, 4]

def display_obs_pred(title, best_error_score, param,  y_test_true, best_predictions, x_train_scale,TEST_SIZE=0):
    """
    show a chart with an observation and prediction
    """
    plt.title(title + 
              " Prediction Quality: {:.2f} RMSE with {} lags, {} pre-steps, {} seq-steps, {} features, {} batch_size, {} neurons \n \
              State: {}, Shuffle: {} \
              Train size: {}, \
              Test size:{} hours".
             format(best_error_score, param.n_lags, param.n_pre_steps, param.n_seq_steps,param.n_features, param.batch_size, param.n_neurons,
                    param.b_stateful, param.b_shuffle,
                    len(x_train_scale), TEST_SIZE))
    plt.plot(y_test_true, label = 'Observed', color='#006699')
    plt.plot(best_predictions, label='Prediction', color='#ff0066')
    plt.legend(loc='best')
    plt.show()
   