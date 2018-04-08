#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:08:10 2018

@author: gonsoomoon
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Class to proess input data
class elec_consume_data():
   def __init__(self, para):
      self._para = para
      
   def process_input(self):
      para = self._para
      data = np.load(os.path.join(para.data_dir, para.input_data))
      print ("data shape: ", np.shape(data))
      # (140256,) -- this is a series of electricy only for one user duirng four years
      total_num_data = len(data)
      num_data = int(total_num_data * para.num_data_ratio)
      
      # Use when a partial data needs such as a test
      
      data = data[0:num_data]
      print("number of data: ", len(data))      
      
      data = data.reshape(-1,1) # (140256, ) -> (140256,1)
      # scale the data to be in the range (0,1)
      scaler = MinMaxScaler(feature_range=(0,1), copy = False)
      data = scaler.fit_transform(data)
      self._scaler = scaler
      #print("scaled data: \n", data)
      
      # calculate # of sequences
      # For example, if num_timesteps =3 and data = [1,2,3,4,5]
      # all sequences are X = ([1,2,3], [2,3,4) Y = ([4], [5])
      num_sequence = num_data - para.num_timesteps
      
      # transform to 4 inputs -> 1 label format
      X = np.zeros((num_sequence, para.num_timesteps))
      #print("X shape: ", np.shape(X)) # (140256, 20)
      Y = np.zeros((num_sequence, 1)) 
      #print("Y shape: ", np.shape(Y)) # (140256,1)
      
      print("# of sequence: ", num_sequence)
      for i in range(num_sequence):
         X[i] = data[i: i + para.num_timesteps].T
         #X[i] = data[i: i + NUM_TIMESTEPS].T      
         #Y[i] = data[i + NUM_TIMESTEPS + 1]
         Y[i] = data[i + para.num_timesteps]   
         
      # reshpae X to three dimensions (samples, timesteps, features)
      X = np.expand_dims(X, axis = 2)
      print("new X shape: ", np.shape(X)) # (140256, 20, 1)
      
      
      # split into training and test sets (add the extra offsets so
      # we can use batch size of 5)
      
      sp = int(para.train_test_ratio * len(data))
      Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
      print("Xtrain.shape, Ytrain.shape, Xtest.shape, , Ytest.shape")
      print(Xtrain.shape, Ytrain.shape, Xtest.shape,  Ytest.shape)
      # (98179, 20, 1) (42077, 20, 1) (98179, 1) (42077, 1)
      
      if para.stateless == False:
         print("State is True")
         # stateful
         # need to make training and test data to multiple of BATCH_SIZE
         train_size = (Xtrain.shape[0] // para.batch_size) * para.batch_size
         test_size = (Xtest.shape[0] // para.batch_size) * para.batch_size
         Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
         Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
      
   
      
      return Xtrain, Ytrain, Xtest, Ytest
   
      #print("data, \n", data[0:25].T)
      #print("X[0:2], \n", X[0:2])
      #print("Y[0:2[], \n", Y[0:2])
      #data, 
      # [[0.37875314 0.38970776 0.32776504 0.29137096 0.28041635 0.30592859
      #  0.29864978 0.28773156 0.28409215 0.29864978 0.28773156 0.26221931
      #  0.24766168 0.30228919 0.29501037 0.27677694 0.27313753 0.309568
      #  0.29137096 0.27677694 0.27313753 0.30228919 0.28773156 0.28041635
      #  0.28041635]]
      #X[0:2], 
      # [[0.37875314 0.38970776 0.32776504 0.29137096 0.28041635 0.30592859
      #  0.29864978 0.28773156 0.28409215 0.29864978 0.28773156 0.26221931
      #  0.24766168 0.30228919 0.29501037 0.27677694 0.27313753 0.309568
      #  0.29137096 0.27677694]
      # [0.38970776 0.32776504 0.29137096 0.28041635 0.30592859 0.29864978
      #  0.28773156 0.28409215 0.29864978 0.28773156 0.26221931 0.24766168
      #  0.30228919 0.29501037 0.27677694 0.27313753 0.309568   0.29137096
      #  0.27677694 0.27313753]]
      #Y[0:2[], 
      # [[0.30228919]
      # [0.28773156]]

   # Compute an inversed data
   def inverse_scaled_data(self, scaled_data):
      scaler = self._scaler
      data_inversed = scaler.inverse_transform(scaled_data)
      return data_inversed
      
      