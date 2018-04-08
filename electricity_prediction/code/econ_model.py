#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:33:36 2018

@author: gonsoomoon
"""
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json

import math
import os
import numpy as np

class econ_model():
   def __init__(self, parameter, Xtrain, Ytrain, Xtest, Ytest):
      self._para = parameter
      self._Xtrain = Xtrain
      self._Ytrain = Ytrain
      self._Xtest = Xtest
      self._Ytest = Ytest
      model_path, weight_path = self.get_model_saving_path()
      self._model_path = model_path
      self._weight_path = weight_path

   ########################################
   # Get saving path of a model
   ########################################
   def get_model_saving_path(self):
      para = self._para
      if para.stateless:
         model_path = os.path.join(para.ckpt_dir, para.stateless_dir, para.model_arch)
         weight_path = os.path.join(para.ckpt_dir, para.stateless_dir, para.model_weight)
      else:
         model_path = os.path.join(para.ckpt_dir, para.stateful_dir,  para.model_arch)
         weight_path = os.path.join(para.ckpt_dir, para.stateful_dir, para.model_weight)
      return model_path, weight_path


   ###################################################
   # Build a model
   ###################################################
   def build_model(self):
      para = self._para
      if para.stateless:
         model = Sequential()
         model.add(LSTM(para.hidden_size, input_shape=(para.num_timesteps, 1),
                        return_sequences=False))
         model.add(Dense(1))
      else:
         model = Sequential()
         
         model.add(LSTM(para.hidden_size, stateful = True,
                        batch_input_shape=(para.batch_size, para.num_timesteps, 1),
                        return_sequences = False))
         model.add(Dense(1))
      return model
   
   def create_model(self):
      para = self._para
      model_path = self._model_path
      weight_path = self._weight_path
      # if model is reused
      if para.model_weight_reuse:
         model = model_from_json(open(model_path).read())
         model.load_weights(weight_path)
         print("weight loaded from : ", weight_path)
      else:
         model = self.build_model()
         
      #print(model.summary())
      model.compile(loss = "mean_squared_error", optimizer = "adam",
                    metrics = ["mean_squared_error"])
      self._model = model
      return model
   
   
   ###################################################
   # Train a model
   ###################################################
   #model = create_model(MODEL_WEIGHT_REUSE)
   
   def train(self):
      para = self._para
      model = self._model
      Xtrain = self._Xtrain
      Ytrain = self._Ytrain
      
      
      if para.stateless:
         history = model.fit(Xtrain, Ytrain, epochs = para.num_epochs, batch_size = para.batch_size,
                   validation_split= para.train_val_ratio,
                   shuffle=False)
         
      else:
         for i in range(para.num_epochs):
            print("Epoch {:d} / {:d}".format(i+1, para.num_epochs))   

            # To split train into train and validation            
            train_size = int(Xtrain.shape[0] * (1 - para.train_val_ratio))
            Xtrain2, Xval = Xtrain[0:train_size], Xtrain[train_size:]
            Ytrain2, Yval = Ytrain[0:train_size], Ytrain[train_size:]            
            
            # To make a multiple of a batch_size
            train_size = (Xtrain2.shape[0] // para.batch_size) * para.batch_size
            val_size = (Xval.shape[0] // para.batch_size) * para.batch_size
            Xtrain2, Ytrain2 = Xtrain2[0:train_size], Ytrain2[0:train_size]
            Xval, Yval = Xval[0:val_size], Yval[0:val_size]
   
            history = model.fit(Xtrain2, Ytrain2, batch_size=para.batch_size, epochs=1,
                     validation_data=(Xval, Yval),
                     shuffle=False)
            
            model.reset_states()
      self._model = model
      return model, history
   
   def evaluate(self):   
      para = self._para
      model = self._model
      Xtest = self._Xtest
      Ytest = self._Ytest
      
      #score, _ = model.evaluate(Xtest, Ytest, batch_size = para.batch_size)
      score, _ = model.evaluate(Xtest, Ytest, batch_size = para.batch_size)
      
      print("score: \n", score)

      rmse = math.sqrt(score)
      print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))
   
   ########################################
   # Predict
   ########################################
   def predict(self):
      para = self._para
      model = self._model
      Xtest = self._Xtest
      Ytest = self._Ytest

      #predictions = model.predict(Xtest, batch_size=1, verbose=0)
      predictions = model.predict(Xtest, batch_size=para.batch_size, verbose=0)
      #print("Ytest: \n", Ytest)      
      #print("Prediction: \n", predictions)

      return Ytest, predictions
   
   
   
   ########################################
   # Save a model
   ########################################
   def save_model(self):
      model = self._model
      model_path = self._model_path
      weight_path = self._weight_path
      
      model_json = model.to_json()
      open(model_path,'w').write(model_json)
      print("model saved: ", model_path)
      
      
      # Save weight
      model.save_weights(weight_path, overwrite=True)
      print("weight saved: ", weight_path)   

