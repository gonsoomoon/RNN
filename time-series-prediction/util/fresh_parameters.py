#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:37:04 2018

@author: gonsoomoon
"""

# Save all parameters
class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self, n_repeats, n_data_period, n_lags, n_seq_steps, n_pre_steps, n_features, batch_size, forcast_batch_size, test_size, 
                 n_epochs, n_neurons,
                 b_stateful, b_shuffle,
                 ratio_rec_dropout,
                 ratio_input_dropout,
                 ratio_validation,input_folder, input_file_01, debug_folder,
                 b_load_model, model_folder, model_json, model_weight): 
        """
        """
        self.n_repeats = n_repeats    
        self.n_data_period = n_data_period # 0 means 52 wk, 1 means 3 months, 2 means 4 weeks 3 means 1 week and else is 3 days          
        self.n_lags = n_lags # num of lags
        self.n_seq_steps = n_seq_steps
        self.n_pre_steps = n_pre_steps        
        self.n_features = n_features
        self.batch_size = batch_size    
        self.forcast_batch_size = forcast_batch_size
        self.test_size = test_size
        self.n_epochs = n_epochs
        self.n_neurons = n_neurons    
        self.b_stateful = b_stateful
        self.b_shuffle = b_shuffle
        self.ratio_rec_dropout = ratio_rec_dropout
        self.ratio_input_dropout = ratio_input_dropout
        self.ratio_validation = ratio_validation
        self.input_folder = input_folder
        self.input_file_01 = input_file_01
        self.debug_folder = debug_folder
        self.b_load_model = b_load_model
        self.model_folder = model_folder
        self.model_json = model_json
        self.model_weight = model_weight

class mlp_parameters():
    """
    Arguments for data processing.
    """
    def __init__(self, n_repeats, n_data_period, n_lags, n_seq_steps, n_pre_steps, batch_size, test_size, 
                 n_epochs, n_neurons,
                 b_shuffle,
                 ratio_dropout,
                 ratio_validation,input_folder, input_file_01, debug_folder ): 
        """
        """
        self.n_repeats = n_repeats    
        self.n_data_period = n_data_period # 0 means 52 wk, 1 means 3 months, 2 means 4 weeks 3 means 1 week and else is 3 days          
        self.n_lags = n_lags # num of lags
        self.n_seq_steps = n_seq_steps
        self.n_pre_steps = n_pre_steps        
        self.batch_size = batch_size    
        self.test_size = test_size
        self.n_epochs = n_epochs
        self.n_neurons = n_neurons    
        self.b_shuffle = b_shuffle
        self.ratio_dropout = ratio_dropout
        self.ratio_validation = ratio_validation
        self.input_folder = input_folder
        self.input_file_01 = input_file_01
        self.debug_folder = debug_folder
        
        

