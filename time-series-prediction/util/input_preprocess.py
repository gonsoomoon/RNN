#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""
import numpy as np
import pandas as pd
from general_util import save_csv_file


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