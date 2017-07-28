"""
Created on Thu Jul 20 16:21:35 2017

@author: gonsoomoon
"""

from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

class RNN(object):
    def __init__(self, inputFile, lstmLevel, sequence_length, moving_window, epoch):

        # Hyperparameters are set
        self.lstmLevel = lstmLevel
        self.epoch = epoch
        self.sequence_length = sequence_length
        self.moving_window = moving_window
        self.learning_rate = 0.1

        # Attributes are set in the readInput()        
        self.inputFile = inputFile
        self.sentence = ''
        self.data_size = 0
        self.vocabulary_size = 0        
        self.hidden_size = 0
        self.num_classes = 0        
        self.char_set = [] # Unique character
        self.char_dic = [] # (Key, Value) e.g. 'Unique character', index
        
        # Attributes are set in the batchData()                
        self.dataX = []
        self.dataY = []
        self.batch_size = 1        
     
    # Print out attributes
    def verifyAttributes(self):
        print('[Info] inputFile: ', self.inputFile)    
        print('[Info] sentence: ', self.sentence)            
        print('[Info] data_size: ', self.data_size)            
        print('[Info] vocabulary_size: ', self.vocabulary_size)                    
        print('[Info] hidden_size: ', self.hidden_size)                    
        print('[Info] num_classes: ', self.num_classes)                            
        print('[Info] char_set: ', self.char_set)            
        print('[Info] char_dic: ', self.char_dic)                    
        print('[Info] sequence_length: ', self.sequence_length)                            
        print('[Info] moving window: ', self.moving_window)                                    
        print('[Info] epoch: ', self.epoch)                                            
        print('[Info] lstm level: ', self.lstmLevel)                                                    
        print ('[Info] batch size: %d' % (self.batch_size))         
    
    
    
    def run(self):
        # read text from input file and the related attributes are set
        self.readInput()
        # make batch data from the text and the batch_size is set        
        self.batchData()
        # Print out attributes
        self.verifyAttributes()        
        # Train and Predict
        self.train()              

        
    # Read input from the file        
    def readInput(self):
        self.sentence = open(self.inputFile,'r').read()
        self.char_set = list(set(self.sentence))
        self.char_dic = {w:i for i, w in enumerate(self.char_set)}        
        self.data_size = len(self.sentence)
        self.vocabulary_size = len(self.char_set)
        self.hidden_size = self.vocabulary_size
        self.num_classes = self.vocabulary_size        
        
    # Split the sentence by the sequence length into the dataX
    # for a staring point, move by the number of the moving window
    def batchData(self):
        dataX = []
        dataY = []
    
        i =0    
        #for i range(0, len(sentence) - sequence_length):
        while (i + self.sequence_length) < len(self.sentence):
            #print ('i + self.sequence_length ', i + self.sequence_length, 'len(sentence): ', len(self.sentence))
            x_str = self.sentence[i : i + self.sequence_length]
            y_str = self.sentence[i+1 : i + self.sequence_length + 1]
            print('[Info] Input and Target by batch: ', i, x_str, '->', y_str)
            # In the case of 1 of moving window and 20 of sequence length
            # 0 if you want to build a ship, d -> f you want to build a ship, do
            # 1 f you want to build a ship, do ->  you want to build a ship, don
            # 2  you want to build a ship, don -> you want to build a ship, don'
        
            
            x = [self.char_dic[c] for c in x_str] # x str to index
            y = [self.char_dic[c] for c in y_str] # y str to index
            
            dataX.append(x)
            dataY.append(y)
            i = i + self.moving_window
            
        self.dataX = dataX
        self.dataY = dataY
        self.batch_size = len(dataX)        


    # Build, train and run tensorfolw graph    
    def train(self):        
        #==============================================
        # Build LSTM and FC graph
        # 1. Define X, Y placeholder
        # 2. Change X to X_one_hot (one hot encoding)
        # 3. Define basic LSTM with hidden_size that is # of neurons
        # 4. Stack LSTMs
        # 5. Run the stacked LSTMs with the input, x_one_hot
        # 6. Reshape outputs from the stacked LSTM to feed in the FC layer
        # 7. Run the FC with the input, X_for_fc, the output, num_classes and 
        #    no activation function
        #==============================================
        X = tf.placeholder(tf.int32, [None, self.sequence_length])
        Y = tf.placeholder(tf.int32, [None, self.sequence_length])
        
        # One-hot encoding
        X_one_hot = tf.one_hot(X, self.num_classes)
        # print(X_one_hot) # check out the shape
        
        
        multi_cells = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.lstmLevel)], state_is_tuple = True)
        #multi_cells = rnn.MultiRNNCell([self.lstm_cell() for _ in range(1)], state_is_tuple = True)        
        # outputs: unfolding size * hidden size, state = hidden size
        outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype = tf.float32)
        
        # FC layer
        X_for_fc = tf.reshape(outputs, [-1, self.hidden_size])
        outputs = tf.contrib.layers.fully_connected(X_for_fc,
                                                    self.num_classes,
                                                    activation_fn = None)
        
        #==============================================
        # Build loss function graph
        # 1. Reshape the output from FC to be fitting for the sequence loss function
        # 2. Define weights
        # 3. Define sequence_loss with logits with the outputs, targets with Y and weights with weights
        # 4. Define mean loss
        # 5. Define AdamOptimizer
        #==============================================
        
        # reshape out for sequence_loss
        outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.num_classes])
        
        # All weights are 1 ( equal weights)
        weights = tf.ones([self.batch_size, self.sequence_length])
        
        
        sequence_loss = tf.contrib.seq2seq.sequence_loss(
                logits = outputs,
                targets = Y,
                weights = weights)
        mean_loss = tf.reduce_mean(sequence_loss)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # For tensorboard
        mean_loss_tb = tf.summary.scalar("Mean loss", mean_loss)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(mean_loss)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # For tensorboard graph
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs')
        writer.add_graph(sess.graph)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        

        #==============================================
        # Train the graph, updating weights, computing loss and then generating outputs 
        #==============================================

        print ('==================Loss value==================')        
        for i in range(self.epoch):
            _, l, results = sess.run(
                       [train_op, mean_loss, outputs],
                    feed_dict= {X: self.dataX, Y: self.dataY})
        

            print(i,' loss: ', l)
            
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                #print ('index: ', index)
                #print(i, j, ''.join([char_set[t] for t in index]), l)
                
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # Build tensorboard Graph            
            s = sess.run(summary, feed_dict={X: self.dataX, Y: self.dataY})
            writer.add_summary(s, global_step= i)
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$                             


        #==============================================
        # Test data on the model after the training
        #==============================================
        print ('==================Prediction==================')        
        results = sess.run(outputs, feed_dict={X: self.dataX})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            
            if j is 0: # print all for the first result to make a sentence
                print(''.join([self.char_set[t] for t in index]), end='')        
            else:
                # From the last to size of moving_window
                lastIndex = index[len(index) - self.moving_window :]
                print(''.join([self.char_set[t] for t in lastIndex]), end='')
            
            
        print ('\n')  
        sess.close()

                
        return results
        

    # Make a LSTM cell with hidden_size (each unit output vector size)
    def lstm_cell(self):
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
        return cell
    
        
    
            
# Test function

def test(inputFile,lstmLevel, sequence_length, moving_window, epoch):
    # Create rnn object
    rnn = RNN(inputFile,lstmLevel, sequence_length, moving_window, epoch)
    rnn.run()

    
if __name__ == "__main__":
    # Read command line arguments for hyperparameters    
    inputFile = sys.argv[1]
    lstmLevel = int(sys.argv[2])
    sequence_length = int(sys.argv[3])
    moving_window = int(sys. argv[4])
    epoch = int(sys.argv[5])
    
    
    test(inputFile,lstmLevel, sequence_length, moving_window, epoch)                    
 

