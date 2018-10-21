#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
import numpy as np
import pandas as pd
from fresh_general_util import save_csv_file
from sklearn.preprocessing import LabelEncoder


def load_input_personid(input_file_path):
    """
    Load an electricity spending and person id
    """
    series_raw = pd.read_csv(input_file_path, encoding = 'CP949', parse_dates=[0], index_col='date')
    series_raw.columns = ['elect_sp', 'person_id']
    return series_raw
 
# Load data from csv
def load_input(input_file_path):
   series_raw = pd.read_csv(input_file_path, encoding = 'CP949', parse_dates=[0], index_col='date')
   series_raw.columns = ['elect_sp']
   return series_raw

from datetime import date
# depending on DATA_PERIOD, the data set is selected
def get_sample_period(DATA_PERIOD):
    """
    Depending on a number of period, a start date and an end date are determined
    """
    if DATA_PERIOD == 0:
        # 52 wk
        start_date = date(2011,1,3) # Monday
        end_date = date(2011,12,25) # Sunday    
    elif DATA_PERIOD == 1:
        # 3 months
        start_date = date(2011,1,3) # Monday
        end_date = date(2011,3,27) # Sunday
    elif DATA_PERIOD == 2:
        # 4 weeks
        start_date = date(2011,1,3) # Monday
        end_date = date(2011,1,30) # Sunday    
    elif DATA_PERIOD == 3:
        start_date = date(2011,1,3) # Monday
        end_date = date(2011,1,9) # Sunday
    else:
        start_date = date(2011,1,3) # Monday
        end_date = date(2011,1,5) # Sunday
    return start_date, end_date

# Sample a period of date
def sample_dataset(start_date, end_date, series, DEBUG_FOLDER, verbose=False):
    # Make 52 weeks data set, filling hours on empty hour-slots
    df_sample = preprocess_fill_hour(series, start_date, end_date)
    df_sample = df_sample.astype('float32')
    if verbose:
        file_name = DEBUG_FOLDER + str((end_date - start_date).days) + ".csv"
        save_csv_file(df_sample, filename=file_name)    
    return df_sample

def split_X_Y_2(train_scale, val_scale, test_scale, param, y_index, verbose=False):
    """
    y_index means a position so that y positions calumates a multiple of y_index (if 2, 0, 2, 4, 6 ...)
    """
    # split into input and outputs
    n_obs = param.n_lags * param.n_features
    y_index = 2 # 0, 2, 4, 6, 
    np_train_scale = np.array(train_scale)
    np_val_scale = np.array(val_scale)
    np_test_scale = np.array(test_scale)
    
    train_X = np_train_scale[:, :n_obs]
    train_y = get_y(np_train_scale[:,n_obs:], param.n_pre_steps, y_index)
    train_y = np.array(train_y).T
    val_X = np_val_scale[:, :n_obs]
    val_y = get_y(np_val_scale[:,n_obs:], param.n_pre_steps, y_index)
    val_y = np.array(val_y).T
    test_X = np_test_scale[:, :n_obs]
    test_y = get_y(np_test_scale[:,n_obs:], param.n_pre_steps, y_index)
    test_y = np.array(test_y).T


    if verbose:
        print("Train shape X: {} Y: {}".format(train_X.shape, train_y.shape))
        print("Train_X: {} \n Train_Y: {}".format(train_X[0,:], train_y[0,:]))
        print("Val shape X: {} Y: {}".format(val_X.shape, val_y.shape))
        print("Test shape X: {} Y: {}".format(test_X.shape, test_y.shape))
        print("Test shape X: {} Y: {}".format(test_X.shape, test_y.shape))
        #print("Y_test: {}".format(y_test.shape))
        
    return train_X, train_y, val_X, val_y, test_X, test_y

def split_X_Y(data, lags, kind = "train", DEBUG_FOLDER ="." , verbose=False):
    """
    For one feature, split X and Y
    """
    if type(data) is pd.DataFrame:
        supervised_np = np.array(data)
    else:
        supervised_np = data
    X, Y = supervised_np[:, 0:lags], supervised_np[:, lags:]
    
    if verbose:
        file_name = DEBUG_FOLDER + "X_" + kind + "csv"
        save_csv_file(X, filename=file_name)
    return X, Y
 
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in = 1, n_out = 1, dropnan=True):
    """
    Make a format of X and Y with a number of lags and a number of predicting steps
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    agg = concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
    return agg   
 
def preprocess_fill_hour(df_raw, start_date, end_date):
    """
    Fill an hour with 0 for missing hours
    """
    df_wk = df_raw[df_raw.index.date >= start_date]
    df_wk = df_wk[df_wk.index.date <= end_date]
    return df_wk   
 
import math
from sklearn.preprocessing import MinMaxScaler


def prepare_data3(series, param ,verbose=False):
    """
    On the input, return train, val and test data set
    """
    raw_values = series.values
    
    # encoding id column
    encoder = LabelEncoder()
    raw_values[:,1] = encoder.fit_transform(raw_values[:,1])
    raw_values = raw_values.astype('float32')
    ##### scale
    scaler = MinMaxScaler(feature_range=(-1,1))
    # scale each column of the raw_values
    scaled_values = scaler.fit_transform(raw_values)
    
    ##### make a supervised form
    supervised = series_to_supervised(scaled_values, param.n_lags, param.n_pre_steps, dropnan=True)
    supervised_values_scale = supervised.values
    ##### Split into train and test
    # Get an index of test data set
    test_start_idx = param.test_size -(param.n_pre_steps -1)

    # Split into a whole train and test in the form of being scaled and raw
    train_whole_scale = supervised_values_scale[: -test_start_idx]
    train_whole_raw = raw_values[: -test_start_idx]    
    test_scale = supervised_values_scale[-test_start_idx :]
    test_raw = raw_values[-param.test_size :]    
    
    ##### Split the train into train and val data set
    n_multiple_train_whole = math.floor(train_whole_scale.shape[0] / param.batch_size)
    n_multiple_train = math.ceil(n_multiple_train_whole * (1 - param.ratio_validation)) # if it is 3.6, --> 4
    n_multiple_val = n_multiple_train_whole - n_multiple_train
    
    train_end_idx = n_multiple_train * param.batch_size
    val_end_idx = n_multiple_train_whole * param.batch_size
    train_scale, val_scale = train_whole_scale[0:train_end_idx], train_whole_scale[train_end_idx: val_end_idx]
    train_raw, val_raw = train_whole_raw[0:train_end_idx], train_whole_raw[train_end_idx:]
    
    ##### Get a train data by a multiple of a batch size
    train_len = train_scale.shape[0]
    # Get a multiple of a batch size
    train_end_idx = int(train_len/ param.batch_size) * param.batch_size
    train_scale_multiple = train_scale[0:train_end_idx, :]
    train_raw_multiple = train_raw[0:train_end_idx, :]    
    
    ##### Get a validation data by a multiple of a batch size
    val_len = val_scale.shape[0]
    # Get a multiple of a batch size
    val_end_idx = int(val_len/param.batch_size) * param.batch_size
    val_scale_multiple = val_scale[0:val_end_idx, :]
    val_raw_multiple = val_raw[0:val_end_idx, :]    
        
    if verbose:
        print("data sample after LabelEncoder: \n",raw_values[0:1,:])    
        print("shape of supervised_values_scale: \n", supervised_values_scale.shape)    
        print("supervised_values_scale: \n", supervised_values_scale[0:1,:])
        print("Shape of train_scale: ", train_scale.shape)
        print("Shape of train_scale_multiple: ", train_scale_multiple.shape)        
        print("Shape of train_raw_multiple: ", train_raw_multiple.shape)                
        print("Shape of val_scale: ", val_scale.shape)
        print("Shape of val_scale_multiple: ", val_scale_multiple.shape)                
        print("Shape of val_raw_multiple: ", val_raw_multiple.shape)                        
        print("Shape of test_scale: ", test_scale.shape)
        print("Shape of test_raw: ", test_raw.shape)    

    return scaler, train_scale_multiple, val_scale_multiple, test_scale

def prepare_data(series, param ,verbose=False):
    """
    Scale, make data as a form of X and Y, split X and Y, finally make size as a multiple of a btach size 
    """
    raw_values = series.values
    
    # scale
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    
    ##### make a supervised form
    supervised = series_to_supervised(scaled_values, param.n_lags, param.n_pre_steps, dropnan=True)
    supervised_values_scale = supervised.values
    ##### Split into train and test
    # Get an index of test data set
    test_start_idx = param.test_size -(param.n_pre_steps -1)

    
    # Split into a whole train and test in the form of being scaled and raw
    train_whole_scale = supervised_values_scale[: -test_start_idx]
    train_whole_raw = raw_values[: -test_start_idx]    
    test_scale = supervised_values_scale[-test_start_idx :]
    test_raw = raw_values[-param.test_size :]    

    
    ##### Split the train into train and val data set
    train_end_idx = math.ceil((train_whole_scale.shape[0] * (1 - param.ratio_validation)))
    train_scale, val_scale = train_whole_scale[0:train_end_idx], train_whole_scale[train_end_idx:]
    train_raw, val_raw = train_whole_raw[0:train_end_idx], train_whole_raw[train_end_idx:]
    
    ##### Get a train data by a multiple of a batch size
    train_len = train_scale.shape[0]
    # Get a multiple of a batch size
    train_end_idx = int(train_len/ param.batch_size) * param.batch_size
    train_scale_multiple = train_scale[0:train_end_idx, :]
    train_raw_multiple = train_raw[0:train_end_idx, :]    
    
    ##### Get a validation data by a multiple of a batch size
    val_len = val_scale.shape[0]
    # Get a multiple of a batch size
    val_end_idx = int(val_len/param.batch_size) * param.batch_size
    val_scale_multiple = val_scale[0:val_end_idx, :]
    val_raw_multiple = val_raw[0:val_end_idx, :]    
    
    ##### Get a train and test raw data
    #train_whole_raw_end = -n_test - n_pred + 1
    #train_whole_raw, test_raw = raw_values[n_lag : train_whole_raw_end], raw_values[-n_test:]
    #train_whole_raw_multiple = train_whole_raw[0:train_whole_raw_end, :]
    
    if verbose:
        print("Shape of train_scale: ", train_scale.shape)
        print("Shape of train_scale_multiple: ", train_scale_multiple.shape)        
        print("Shape of train_raw_multiple: ", train_raw_multiple.shape)                
        print("Shape of val_scale: ", val_scale.shape)
        print("Shape of val_scale_multiple: ", val_scale_multiple.shape)                
        print("Shape of val_raw_multiple: ", val_raw_multiple.shape)                        
        print("Shape of test_scale: ", test_scale.shape)
        print("Shape of test_raw: ", test_raw.shape)        
    
    return scaler, train_scale_multiple, train_raw_multiple, val_scale_multiple, val_raw_multiple, test_scale, test_raw

# Retrieve a multiple of batch size for train and validation data
def prepare_data2(series, param ,verbose=False):
    raw_values = series.values
    
    # scale
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    
    ##### make a supervised form
    supervised = series_to_supervised(scaled_values, param.n_lags, param.n_pre_steps, dropnan=True)
    supervised_values_scale = supervised.values
    ##### Split into train and test
    # Get an index of test data set
    test_start_idx = param.test_size -(param.n_pre_steps -1)

    
    # Split into a whole train and test in the form of being scaled and raw
    train_whole_scale = supervised_values_scale[: -test_start_idx]
    train_whole_raw = raw_values[: -test_start_idx]    
    test_scale = supervised_values_scale[-test_start_idx :]
    test_raw = raw_values[-param.test_size :]    

    
    ##### Split the train into train and val data set
    n_multiple_train_whole = math.floor(train_whole_scale.shape[0] / param.batch_size)
    n_multiple_train = math.ceil(n_multiple_train_whole * (1 - param.ratio_validation)) # if it is 3.6, --> 4
#    n_multiple_val = n_multiple_train_whole - n_multiple_train
    
    train_end_idx = n_multiple_train * param.batch_size
    val_end_idx = n_multiple_train_swhole * param.batch_size
    train_scale, val_scale = train_whole_scale[0:train_end_idx], train_whole_scale[train_end_idx: val_end_idx]
    train_raw, val_raw = train_whole_raw[0:train_end_idx], train_whole_raw[train_end_idx:]
    
    ##### Get a train data by a multiple of a batch size
    train_len = train_scale.shape[0]
    # Get a multiple of a batch size
    train_end_idx = int(train_len/ param.batch_size) * param.batch_size
    train_scale_multiple = train_scale[0:train_end_idx, :]
    train_raw_multiple = train_raw[0:train_end_idx, :]    
    
    ##### Get a validation data by a multiple of a batch size
    val_len = val_scale.shape[0]
    # Get a multiple of a batch size
    val_end_idx = int(val_len/param.batch_size) * param.batch_size
    val_scale_multiple = val_scale[0:val_end_idx, :]
    val_raw_multiple = val_raw[0:val_end_idx, :]    
    
    
    if verbose:
        print("Shape of train_scale: ", train_scale.shape)
        print("Shape of train_scale_multiple: ", train_scale_multiple.shape)        
        print("Shape of train_raw_multiple: ", train_raw_multiple.shape)                
        print("Shape of val_scale: ", val_scale.shape)
        print("Shape of val_scale_multiple: ", val_scale_multiple.shape)                
        print("Shape of val_raw_multiple: ", val_raw_multiple.shape)                        
        print("Shape of test_scale: ", test_scale.shape)
        print("Shape of test_raw: ", test_raw.shape)        
    
    return scaler, train_scale_multiple, train_raw_multiple, val_scale_multiple, val_raw_multiple, test_scale, test_raw
 
def get_y(data, n_outputs, y_index):
    """
    Extract y value from a feature set
    """
    y_outputs = list()
    for i in range(n_outputs):
        index = y_index * i
        #print("index: ", index)
        y = data[:, index]
        y_outputs.append(y)
        #print(i)
    return y_outputs   
   