#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:13:20 2017

@author: gonsoomoon
"""

print("===== vocabulary.py ===========")

import os
#"basedir = "/Users/gonsoomoon/Documents/DeepLearning/rnn/Attentional-Interfaces-O-Reilly-master"

import warnings
warnings.filterwarnings('ignore')

#import argparse
#import json
#import os
import pickle
#import re
#import tensorflow as tf
#import numpy as np

"""
from unidecode import (
        unidecode,
)


from random import (
        shuffle,
)

from tqdm import (
        tqdm,
)

from collections import (
        Counter,
)
"""

UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

class Vocab():
    """
    Class for processing tokens to ids and vice versa
    """
    def __init__(self, vocab_file, 
                 max_vocab_size=75133, 
                 verbose=True):
        """
        """
        self.verbose = verbose
        self._token_to_id = {}
        self._id_to_token = {}
        self._size = -1

        """
        vocab_file format
        
        <unk> 0
        <pad> 0
        the 336693
        . 327192
        , 276280
        and 164140
        """
        
        with open(vocab_file, 'rt', encoding = 'utf-8') as f:

            for line in f:
                tokens = line.split()                
                
                # White space in vocab file (' ':<count>)
                if len(tokens) == 1:
                    count = tokens[0]
                    idx = line.index(count)
                    t = line[:idx-1]
                    tokens = (t, count)
                    
                if len(tokens) != 2:
                    continue
                
                if tokens[0] in self._token_to_id:
                    continue

                self._size += 1
                
                if self._size >= max_vocab_size:
                    print ('Too many tokens! > %i\n' % max_vocab_size)
                    break
                

                
                self._token_to_id[tokens[0]] = self._size
                self._id_to_token[self._size] = tokens[0]
                
                
    def __len__(self):
        #return self._size + 1
        return self._size
    
    def token_to_id(self, token):
        if token not in self._token_to_id:
            if self.verbose:
                print("ID not found for  %s" % token)
            return self._token_to_id[UNKNOWN_TOKEN]
        return self._token_to_id[token]
                
    def id_to_token(self, _id):
        if _id not in self._id_to_token:
            if self.verbose:
                print ("Token not found for ID: %i" % _id)
            return UNKNOWN_TOKEN
        return self._id_to_token[_id]
    
def ids_to_tokens (ids_list, vocab):
    """
    Convert a list of ids to tokens.
    Args:
        ids_list: list of ids to convert to tokens.
        vocab: Vocab class object.
    Returns:
        answer: list of tokens that corresponds to ids_list
    """
    answer = []
    for _id in ids_list:
        token = vocab.id_to_token(_id)
        if token == PAD_TOKEN:
            continue
        answer.append(token)
    return answer

#===================================================
# Utility functions
#===================================================
#import os
#import argparse
#import pickle
#import numpy as np
import random
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#%pylab inline

class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        self.data_dir = "data/processed_reviews/train.p"
        
def sample_data(data_path):
    """
    Sample format of the processed data from data.py
    Args:
        data_path: path for train.p | valid.p
    """
    
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)
        
    # choose a random sample
    rand_index = random.randint(0, len(entries))
    
    # Prepare vocab
    vocab_file = os.path.join('','data/processed_reviews/vocab.txt'  )
    vocab = Vocab(vocab_file, verbose = False, max_vocab_size = 80000)
    
    (processed_review, review_seq_len, label) = entries[rand_index]
    
    print (" ==> Processed Review: ", processed_review)
    print (" ==> Review Len: ", review_seq_len)
    print (" ==> Label: ", label)
    print (" ==> See if processed review makes sense:",
           ids_to_tokens(
                   processed_review,
                   vocab = vocab,))
    
    
    
    





























