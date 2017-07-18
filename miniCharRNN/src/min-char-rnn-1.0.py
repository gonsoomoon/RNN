#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:19:34 2017

@author: gonsoomoon
"""

import numpy as np
import matplotlib.pyplot as plt

class RNN(object):
    def __init__(self, vocab_size,hidsize, learning_rate):
        self.vocab_size = vocab_size
        self.h = np.zeros((hidsize , 1))#a [h x 1] hidden state stored from last batch of inputs        
        
        # parameters
        self.W_hh = np.random.randn(hidsize, hidsize) * 0.01 
        self.W_xh = np.random.randn(hidsize, vocab_size) * 0.01
        self.W_hy = np.random.randn(vocab_size, hidsize) * 0.01
        self.b_h = np.zeros((hidsize, 1))
        self.b_y = np.zeros((vocab_size,1))
        
        # the Adagrad gradient update relies upon having a memory 
        # of the sum of squares of dparams
        
        self.adaW_hh = np.zeros_like(self.W_hh)
        self.adaW_xh = np.zeros_like(self.W_xh)
        self.adaW_hy = np.zeros_like(self.W_hy)
        self.adab_h = np.zeros_like(self.b_h)
        self.adab_y = np.zeros_like(self.b_y)
        
        
        self.learning_rate = learning_rate
        
    # Give the RNN a sequence of inputs and outputs (seq_length long)
    # and use them to adjust the internal state
    def train(self, x, y):
        # initialize 
        xhat = {} # holds 1-of-k representation of x
        yhat = {} # holds 1-of-k representation of predicted y, unnormalized log prob.
        p = {} # the normalized prob. of each output through time
        h = {} # holds state vectors through time
        # we will need to access the previous state to calculate the current state 
        h[-1] = np.copy(self.h) 
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(self.h)
        
        # Forward pass
        loss = 0
        for t in range(len(x)):
            xhat[t] = np.zeros((self.vocab_size, 1))
            xhat[t][x[t]] = 1 # 1-of-k representation of x[t]
            
            # Find new hidden state
            h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)

            
            # Find unnormalized log prob. for next chars
            yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y
            # Find probabilities for next chars
            p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))
            # cross-entropy loss
            loss += -np.log(p[t][y[t],0])
            
        #=====backward pass: compute gradients going backwards=====
        for t in reversed(range(len(x))):
            #backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dy = np.copy(p[t])
            dy[y[t]] -= 1

            #find updates for y
            dW_hy += np.dot(dy, h[t].T)
            db_y += dy

            #backprop into h and through tanh nonlinearity
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - h[t]**2) * dh

            #find updates for h
            dW_xh += np.dot(dh_raw, xhat[t].T)
            dW_hh += np.dot(dh_raw, h[t-1].T)
            db_h += dh_raw

            #save dh_next for subsequent iteration
            dh_next = np.dot(self.W_hh.T, dh_raw)

        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)#clip to mitigate exploding gradients

        #update RNN parameters according to Adagrad
        for param, dparam, adaparam in zip([self.W_hh, self.W_xh, self.W_hy, self.b_h, self.b_y], \
                                [dW_hh, dW_xh, dW_hy, db_h, db_y], \
                                [self.adaW_hh, self.adaW_xh, self.adaW_hy, self.adab_h, self.adab_y]):
            adaparam += dparam * dparam
            param += -self.learning_rate * dparam/np.sqrt(adaparam+1e-8)
            
        self.h = h[len(x)-1]
        
        return loss
        
    
    # Let the RNN generate text
    def sample(self, vocab_size, seed, n):
        ndxs = []
        h = self.h
        
        xhat = np.zeros((vocab_size,1))
        xhat[seed] = 1 # transform to 1-of-k
        
        # n is seq_length
        for t in range(n):
            # update the state
            h = np.tanh(np.dot(self.W_xh, xhat) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            # Softmax function
            p = np.exp(y) / np.sum(np.exp(y))
            # p, probability, is weight so that ix is preferred with higher p
            # As parameter update has been applied, ix is more correct output
            ndx = np.random.choice(range(self.vocab_size), p = p.ravel())
            
            xhat = np.zeros((vocab_size,1))
            xhat[ndx] =1
            # ndxs has a list of output characters            
            ndxs.append(ndx)
            
        return ndxs
    
# test
def test(inputFile, seq_length, hidden_size, learning_rate, predictionLen, printThreshold, epoch):
    # Open a text file
    data = open(inputFile,'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print ('data has %d characters, %d unique.' % (data_size, vocab_size))
    
    #make some dictionaries for encoding and decoding from 1-of-k
    ################################
    # data I/O
    # Through unique characters, compute size of vocabulary
    # Make a key-value pair (character of data, index of vocabulary)
    # As an example of "Hello world recurrent neural", 
    # ix_to_char: {0: 'a', 1: ' ', 2: 'c', 3: 'e', 4: 'd', 5: 'H', 6: 'l', 
    # 7: 'o', 8: 'n', 9: 'r', 10: 'u', 11: 't', 12: 'w'}
    ################################
    
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }        
    
    # Create rnn object
    rnn = RNN(vocab_size, hidden_size, learning_rate)
    
    losses = []
    # loss of iteration of 0
    smooth_loss = -np.log(1.0/vocab_size) * seq_length 
    losses.append(smooth_loss)
    
    n, p = 0, 0
    

    while n < epoch:
        if p + seq_length+1 >= data_size or n == 0: 
            rnn.h = np.zeros((hidden_size,1)) # reset RNN memory            
            p = 0 # go from start of data
        
        ################################     
        # Prepare for inputs and targets
        ################################
        # For the input: "Hello world recurrent neural"
        # ix_to_char: {0: 'a', 1: ' ', 2: 'c', 3: 'e', 4: 'd', 5: 'H', 6: 'l', 7: 'o', 8: 'n', 9: 'r', 10: 'u', 11: 't', 12: 'w'}
        # inputs: [5, 3, 6, 6, 7, 1, 12, 7, 9, 6, 4, 1, 9, 3, 2, 10, 9, 9, 3, 8, 11, 1, 8, 3, 10, 9, 0] 
        # --> Hello world recurrent nenura
        # targets: [3, 6, 6, 7, 1, 12, 7, 9, 6, 4, 1, 9, 3, 2, 10, 9, 9, 3, 8, 11, 1, 8, 3, 10, 9, 0, 6]
        # --> ello world recurrent neural
        # hprev: [[ 0.] [ 0.] [ 0.]]
        

        inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length+1]]
        
        ################################     
        # Print out sample text 
        ################################         
        if n % printThreshold == 0:    
            # input[0] is seed value
            # vocab_size, seed, length of text to be generated                        
            sample_ix = rnn.sample(vocab_size,inputs[0], predictionLen)
            # Generate characters mapping index to a character
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            # convert to index to character
            seed = ''.join(ix_to_char[ix] for ix in [inputs[0]])
            print ('Seed, the first character: %s' % (seed))
            print ('Prediction: ----\n %s \n----' % (txt, ))
    
        loss = rnn.train(inputs,targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(smooth_loss)
        
        if n % printThreshold == 0:
            print ('iteration %d, smooth_loss = %f' % (n, smooth_loss))
                    
        p += seq_length # move data pointer
        n += 1 # iteration counter

    # Plot loss
    plt.plot(range(len(losses)), losses, 'b' , label = 'smooth loss')    
    plt.xlabel('# of iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    # Parameters: inputFile, seq_length, hidden_size, learning_rate, predictionLength, printThreshold, epoch
    # For input_alphago.txt from https://en.wikipedia.org/wiki/AlphaGo
    #test('input_alphago.txt', 25, 100, 0.1, 1000, 200, 1000)        
    # For input_mini.txt : Hello world recurrent neural
    test('input_mini.txt', 27, 10, 0.1, 27, 1000, 10000)            
        

            
            
            


