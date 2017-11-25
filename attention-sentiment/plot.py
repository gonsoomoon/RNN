#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:27:35 2017

@author: gonsoomoon

"""

import os
import pickle

class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self, model_name):
        """
        """
        self.ckpt_dir = "data/processed_reviews/ckpt"
        self.model_name = model_name
        #self.model_name = "imdb_model_moon"
        
import matplotlib.pyplot as plt
        
def plot_metrics(FLAGS, basedir):
    """
    Plot the loss and accuracy for train | test
    """
    import seaborn as sns
    
    # Load metrics from file
    metrics_file = os.path.join(basedir, FLAGS.ckpt_dir, 'metrics.p' )
    print("metrics file: ", metrics_file)
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    fig, axes = plt.subplots(nrows = 1, ncols =2, figsize = (20,8))
    
    # Plot results
    axl = axes[0]
    axl.plot(metrics["train_acc"], label = 'train accuracy')
    axl.plot(metrics["valid_acc"], label = 'valid accuracy')
    axl.legend(loc=4)
    axl.set_title('Accuracy')
    axl.set_xlabel('Epoch')
    axl.set_ylabel('train | valid accuracy')
    
    ax2 = axes[1]
    ax2.plot(metrics["train_loss"], label = 'train loss')
    ax2.plot(metrics["valid_loss"], label = 'valid loss')
    ax2.legend(loc=3)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('train | valid loss')
    
    plt.show()
    


        
