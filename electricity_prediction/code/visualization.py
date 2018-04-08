#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:53:18 2018

@author: gonsoomoon
"""

import matplotlib.pyplot as plt     
import pandas as pd 
import numpy as np

class visualization():

   # display train and validation loss
   def display_train_loss(self, history):   
      print(history.history.keys())      
      # summarize history for loss
      print("loss: \n", history.history['loss'] )
      print("val_loss: \n", history.history['val_loss'] )      
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      #plt.legend(['train', 'val'], loc='upper left')
      plt.show()
   
   # display real value and prediction
   def display_elect_pred(self,inversed_pred,inversed_Ytest):
      pred, = plt.plot(inversed_pred, label='Prediction')
      real, = plt.plot(inversed_Ytest, label='Real')   
      plt.title('Electric Consumption Prediction')
      plt.xlabel("Time Period")
      plt.ylabel("Electric Consumption")
      plt.legend(handles = [pred, real])
      plt.show()
      
   # calculate difference between real and prediction
   def calc_diff(self, labels, predictions):
      num_eles = len(labels)
      diff = [abs(i-j) for i, j in zip(labels, predictions)]
      diff_sum = sum(diff)
      diff_mean = diff_sum / num_eles
      
      return diff_sum, diff_mean

   # save real and prediction to csv file
   def save_prediction(self, labels, predictions):      
      Ytest = np.reshape(labels, (labels.shape[0]))
      predictions = np.reshape(predictions, (predictions.shape[0]))      
      
      result = {'RealValue':Ytest, 'Prediction':predictions}
      #print("result: \n", result)
      df_result = pd.DataFrame.from_dict(data = result, orient='index')
      df_result.T.to_csv('../result/result.csv')
      
