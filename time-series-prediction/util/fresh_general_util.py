#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:29:47 2018

@author: gonsoomoon
"""

import pandas as pd
from pandas import Series


import time
def time_start():
    start_time = time.time()
    return start_time

def time_end(start_time):
    end_time = time.time()
    duration = end_time - start_time
    print("start time: ", start_time)
    print("end time: ", end_time)    
    print("Total exectution time (Min): " + str(duration/60))
    
def set_seed():
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    
def save_csv_file(data, filename="../debug/df.csv"):
    if(type(data) is Series):
        df = pd.DataFrame(data.values.reshape(-1,1))
    elif(type(data) is numpy.ndarray):
        df = pd.DataFrame(data)
    elif(type(data) is list):
        df = pd.DataFrame(data)
    elif(type(data) is pd.DataFrame):
        df = data
    else:
        print("data is not either Series or numpy .array")
        return None
    df.to_csv(filename)
    
    