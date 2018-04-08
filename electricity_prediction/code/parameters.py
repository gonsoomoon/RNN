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
    def __init__(self, hidden_size, STATELESS, MODEL_WEIGHT_REUSE, 
                 NEED_TO_TRAIN,
                 NEED_TO_PREDICT, num_data_ratio,
                 train_test_ratio, 
                 num_timesteps, num_epoch, batch_size):
        """
        """
        self.data_dir = "../data"
        self.ckpt_dir = "../ckpt" # locations of model checkpoints
        self.stateful_dir = "stateful" 
        self.stateless_dir = "stateless"
        self.input_data = "LD_250.npy"
        self.model_arch = "energy_architecture.json"
        self.model_weight = 'energy_weights.h5'        
        self.hidden_size = 10 # num hidden units for RNN        
        self.stateless = STATELESS
        self.model_weight_reuse = MODEL_WEIGHT_REUSE
        self.need_to_train = NEED_TO_TRAIN
        self.need_to_predict = NEED_TO_PREDICT
        self.num_data_ratio = num_data_ratio
        self.train_val_ratio = 0.2
        self.train_test_ratio = train_test_ratio
        self.num_timesteps = num_timesteps
        self.num_epochs = num_epoch # num of epochs
        self.batch_size = batch_size
        

