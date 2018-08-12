#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:52:40 2018

@author: gonsoomoon
"""
import matplotlib.pyplot as plt
import seaborn;seaborn.set()
plt.rcParams["figure.figsize"] = [12, 4]


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