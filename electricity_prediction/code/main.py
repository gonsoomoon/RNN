#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:17:03 2018

@author: gonsoomoon
"""

#import econs_stateful as econs

import parameters # parameter 
import input_data # process input data
import econ_model # create a model
import visualization # display charts

def run(para):
   
   # Create object for an input
   econ_data = input_data.elec_consume_data(para)
   # Based on the parameter, split data into train and test data
   Xtrain, Ytrain, Xtest, Ytest = econ_data.process_input()
   
   # Create object for an model
   e_model = econ_model.econ_model(para, Xtrain, Ytrain, Xtest, Ytest)

   # Create a model
   model = e_model.create_model()
   model.summary()

   # Create a chart object
   chart = visualization.visualization()
   
   # Train and evaluate a model
   if para.need_to_train:
      _, history = e_model.train() # train a model
      chart.display_train_loss(history) # show train and validaton loss

      e_model.save_model() # save a model and weight
      e_model.evaluate() # evaluate with the metic MSR using test data
   
   # Predict
   if para.need_to_predict:
      Ytest, prediction = e_model.predict() # return real and predication
      print("prediction shape: ", prediction.shape)
      
      # Inverse the original value
      inversed_pred = econ_data.inverse_scaled_data(prediction)
      inversed_Ytest = econ_data.inverse_scaled_data(Ytest)   
      # Show sum and mean of difference between inversed real value and prediction
      diff_sum, diff_mean = chart.calc_diff(inversed_Ytest, inversed_pred)
      print("diff_sum, diff_mean: ", diff_sum, diff_mean)
      # Save the two values to csv file
      chart.save_prediction(inversed_Ytest, inversed_pred)
      # Display a chart of them
      chart.display_elect_pred(inversed_pred,inversed_Ytest )
            
      

###################################################
# entry point
###################################################

# Stateful for training
para1 = parameters.parameters(hidden_size = 10,
                             STATELESS = True, 
                             MODEL_WEIGHT_REUSE = False, 
                             NEED_TO_TRAIN = True,
                             NEED_TO_PREDICT = False,
                             train_test_ratio = 0.995, 
                             num_data_ratio = 1.0, # how much of data is used
                             num_timesteps = 5, 
                             num_epoch = 1, 
                             batch_size = 96
                             )

# Stateful for prediction
para2 = parameters.parameters(hidden_size = 10,
                             STATELESS = True, 
                             MODEL_WEIGHT_REUSE = True, 
                             NEED_TO_TRAIN = False,
                             NEED_TO_PREDICT = True,
                             train_test_ratio = 0.995, 
                             num_data_ratio = 1.0,                             
                             num_timesteps = 5, 
                             num_epoch = 1, 
                             batch_size = 96
                             )
# Stateful for training
para3 = parameters.parameters(hidden_size = 10,
                             STATELESS = False, 
                             MODEL_WEIGHT_REUSE = False, 
                             NEED_TO_TRAIN = True,
                             NEED_TO_PREDICT = False,
                             train_test_ratio = 0.995, 
                             num_data_ratio = 1.0, # how much of data is used
                             num_timesteps = 5, 
                             num_epoch = 1, 
                             batch_size = 96
                             )

# Stateful for prediction
para4 = parameters.parameters(hidden_size = 10,
                             STATELESS = False, 
                             MODEL_WEIGHT_REUSE = True, 
                             NEED_TO_TRAIN = False,
                             NEED_TO_PREDICT = True,
                             train_test_ratio = 0.995, 
                             num_data_ratio = 1.0,                             
                             num_timesteps = 5, 
                             num_epoch = 1, 
                             batch_size = 96
                             )


#run(para1) # Train stateless
#run(para2) # Predict stateless
run(para3) # train stateful
run(para4) # Predict stateful
