#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:29:14 2017

@author: gonsoomoon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:01:43 2017

@author: gonsoomoon
"""


import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

print("===== attn_history.py ===========")


class parameters():
    """
    Arguments for data processing
    """
    def __init__(self, model_name, sample_num):
        self.data_dir = "data/processed_reviews"
        self.basedir = ""
        self.ckpt_dir = "data/processed_reviews/ckpt"
        self.attn_history_file = "attn_history_epoch20.p"
        self.model_name = model_name
        self.sample_num = sample_num
        self.num_rows = 5
   
import matplotlib.ticker as ticker
     
def plot_attn(input_sentence, attentions, num_rows, save_loc=None):
    """
    Plot the attention scores.
    Args:
        input_sentence: input sentence (tokens) without <pad>
        attentions: attention scores for each token in input_sentence
        num_rows: how many rows you want the figure to have (we will add 1)
        save_loc: fig will be saved to this location
    """     
    # Determine how many words per row
    words_per_row = (len(input_sentence.split(' ')) // num_rows)
    # Use one extra row in case of remained for quotient above
    #fig, axes = plt.subplots(nrows = num_rows+1, ncols=1, figsize(20,10))
    fig, axes = plt.subplots(nrows = num_rows+1, ncols=1)
    for row_num, ax in enumerate(axes.flat):
        # Isolate pertinent part of sentence and attention scores
        start_index = row_num * words_per_row
        end_index = (row_num * words_per_row) + words_per_row
        _input_sentence = \
            input_sentence.split(' ')[start_index:end_index]
            

        _attentions = np.reshape(
                attentions[0, start_index:end_index],
                (1, len(attentions[0, start_index:end_index])))
        
        #print("attentions: ", _attentions)    
        
        # Plot attn scores (constrained to (0.9, 1) for emphasis)
        im = ax.imshow(_attentions, cmap='Blues', vmin=0.95, vmax=1)
        
        # Set up axes
        ax.set_xticklabels(
                [''] + _input_sentence,
                rotation = 90,
                minor = False,)
        ax.set_yticklabels([''])
        
        # Set x tick to top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='black')
        
        # Show corresponding words at the ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
    # Add color bar
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    
    # display color bar
    cb = fig.colorbar(im, cax = cbar)
    cb.set_ticks([]) # Clean color bar
    
    if save_loc is None:
        plt.show()
    else:
        fig.savefig(save_loc, dpi = fig.dpi, bbox_inches = 'tight')
        
import vocabulary  as voc
        
def get_attn_inputs(FLAGS, review, review_len, raw_attn_scores):
    """
    Return the inputs needed to plot the attn scores.
    These include input_sentence and attn_scores.
    Args:
        FLAGS: parameters
        review: list of ids
        review_len: len of the relevant review
    Return:
        input_sentence: inputs as tokens (words) on len < review_len>
        plot_attn_scores: (1, review_len) shaped scores
    """
    
    review_len = 300
    
    # Data paths
    vocab_path = os.path.join(FLAGS.basedir, 'data/processed_reviews/vocab.txt')
    vocab = voc.Vocab(vocab_path)
    review = review[:review_len]
    attn_scores = raw_attn_scores[:review_len]
    
    # Process input_sentence
    input_sentence = ' '.join([item for item in voc.ids_to_tokens(review, vocab)])
    
    #print("input_sentence: ", input_sentence)        
    #print("attn_scores: ", attn_scores)            
    #print("attn_scores shape: ", attn_scores.shape)                
    #print("attn_scores type: ", type(attn_scores))                    
    
    # Process attn scores (normalize scores between [0,1])
    min_attn_score = min(attn_scores)
    max_attn_score = max(attn_scores)
    normalized_attn_scores = ((attn_scores - min_attn_score) /  \
                              (max_attn_score - min_attn_score))
    
    #print("normalized_attn_scores: ", normalized_attn_scores)                
    
    # Reshape attn scores for plotting
    # change one dimention(300,) to two dimenstions (1,300)
    plot_attn_scores = np.zeros((1, review_len))
    for i, score in enumerate(normalized_attn_scores):
        plot_attn_scores[0,i] = score
        
    #print("plot_attn_scores: ", plot_attn_scores)                
    #print("plot_attn_scores shape: ", plot_attn_scores.shape)                    
    #print("plot_attn_scores dim: ", plot_attn_scores.dim)                        
    
        
    return input_sentence, plot_attn_scores


def process_sample_attn(FLAGS):
    """
    Use plot_attn from utils.py to visualize
    the attention scores for a particular sample FLAGS.sample_num
    """
    
    # Load the attn history
    attn_history_path = os.path.join(
            FLAGS.basedir, FLAGS.ckpt_dir, FLAGS.attn_history_file)
    
    print("attn_history_path: ", attn_history_path)
    with open(attn_history_path, 'rb') as f:
        attn_history = pickle.load(f)
                
    for row in attn_history:
        print("row: ", row, "\t", attn_history[row]["label"], "\t")
    
        
    # Process the history to get the right sample
    sample = "sample_%i" % (FLAGS.sample_num)
    review_len = attn_history[sample]["review_len"]
    review = attn_history[sample]["review"]
    label = attn_history[sample]["label"]
    pred_label = attn_history[sample]["pred_label"]    
    print("pred_label: ", pred_label, " label: ", label)
    attn_scores = attn_history[sample]["attn_scores"][-1]
    
    #print("review_len: ", review_len)
    #print("review: ", review)
    #print("label: ", label)    
    #print("attn_scores: ", attn_scores)    
    
    
    input_sentence, plot_attn_scores = get_attn_inputs(
            FLAGS = FLAGS,
            review = review,
            review_len = review_len,
            raw_attn_scores = attn_scores,)
    
    print("input_sentence: ", input_sentence)        
    #print("plot_attn_scores: ", plot_attn_scores)            
    
    
    # Plot and save fig
    fig_name = "sample_%i" % (FLAGS.sample_num)
    save_loc = os.path.join(FLAGS.basedir, FLAGS.ckpt_dir, fig_name)
    plot_attn(
            input_sentence = input_sentence,
            attentions = plot_attn_scores,
            num_rows = FLAGS.num_rows,
            save_loc = None,)
    
        
    
    
    
        
    