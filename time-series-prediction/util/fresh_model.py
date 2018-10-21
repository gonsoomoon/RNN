#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:52:40 2018

@author: gonsoomoon
"""
import matplotlib.pyplot as plt
import seaborn;seaborn.set()
plt.rcParams["figure.figsize"] = [12, 4]

# Save a model
from keras.models import model_from_json
def save_model(model, param):
   model_json = model.to_json()
   model_file_path =  param.model_json
   model_weight_path =  param.model_weight

   model_file_path = param.model_folder + param.model_json
   model_weight_path = param.model_folder + param.model_weight
   
   with open(model_file_path, "w") as json_file:
      json_file.write(model_json)
   model.save_weights(model_weight_path)
   print("Saved model to disk")
   
# Load a model
def load_model(param):
   model_file_path = param.model_folder + param.model_json   
   model_weight_path = param.model_folder + param.model_weight   
   json_file = open(model_file_path)
   loaded_model_json = json_file.read()
   json_file.close()
   loaded_model = model_from_json(loaded_model_json)
   # load weights into new model
   loaded_model.load_weights(model_weight_path)
   loaded_model.compile(loss='mean_squared_error', optimizer='adam')
   
   return loaded_model
   
      
   

# During a learing process, show loss values with a train and validation data
def display_loss_train(history_list):
    # list all data in history
    #print("length of history_list: ", len(history_list))  
    
    train_loss = list()
    validation_loss = list()
    for history in history_list:
        #print("history.history: ", history.history)
        train_loss.append(history.history['loss'])
        validation_loss.append(history.history['val_loss'])
        
    # summarize history for accuracy

    plt.plot(train_loss, color='blue')
    plt.plot(validation_loss,color='red')
    plt.title('model accuracy')
    plt.ylabel('mse accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()    