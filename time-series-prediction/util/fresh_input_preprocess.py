#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
import numpy as np
import pandas as pd
from fresh_general_util import save_csv_file


# Load data from csv
def load_input(input_file_path):
   series_raw = pd.read_csv(input_file_path, encoding = 'CP949', parse_dates=[0], index_col='date')
   series_raw.columns = ['elect_sp']
   return series_raw

from datetime import date
# depending on DATA_PERIOD, the data set is selected
def get_sample_period(DATA_PERIOD):
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

def split_X_Y(data, lags, kind = "train", DEBUG_FOLDER ="." , verbose=False):
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
    df_wk = df_raw[df_raw.index.date >= start_date]
    df_wk = df_wk[df_wk.index.date <= end_date]
    return df_wk   
 
import math
from sklearn.preprocessing import MinMaxScaler
def prepare_data(series, param ,verbose=False):
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
        print("Shape of train_scale: ", train_scale.shape)
        print("Shape of train_scale_multiple: ", train_scale_multiple.shape)        
        print("Shape of train_raw_multiple: ", train_raw_multiple.shape)                
        print("Shape of val_scale: ", val_scale.shape)
        print("Shape of val_scale_multiple: ", val_scale_multiple.shape)                
        print("Shape of val_raw_multiple: ", val_raw_multiple.shape)                        
        print("Shape of test_scale: ", test_scale.shape)
        print("Shape of test_raw: ", test_raw.shape)        
    
    return scaler, train_scale_multiple, train_raw_multiple, val_scale_multiple, val_raw_multiple, test_scale, test_raw
   