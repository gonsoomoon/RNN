#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:24:43 2017

@author: gonsoomoon
Reference: The book, Learning Tensorflow, https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow
"""
import numpy as np
import tensorflow as tf

#########################################
# Hyperparameter
#########################################
batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
times_step = 6
training_steps = 300
element_size = 1
input_num = 10000


#########################################
# Input data creation
#########################################
digit_to_word_map = {1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five",
                     6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}
digit_to_word_map[0] = "PAD"

even_sentences = []
odd_sentences = []
seqlens = []


# Create input
for i in range(input_num):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1,10,2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2,10,2), rand_seq_len)
    
    if rand_seq_len < times_step: # time_step: 6
        rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))
        
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))

 
data = even_sentences + odd_sentences

# Because of even + odd
seqlens *= 2

#print ("even_sentences: ", even_sentences)
#print ("odd_sentences: ", odd_sentences)
#print ("data: ", data)
print ("length of seqlens: ", len(seqlens))
#data:  ['One Three Nine PAD PAD PAD', 'Seven Nine Seven One One Six', 
#       'Three One Three One PAD PAD', 'Six Four Six PAD PAD PAD', 
#       'Six Six Eight Eight Eight Eight', 'Six Two Eight Six PAD PAD']
#seqlens:  [3, 6, 4, 3, 6, 4]

labels = [1] * input_num + [0] * input_num
#print("labels: ", labels)
# labels:  [1, 1, 1, 0, 0, 0]

# Make labels a form of one-hot encoding
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2 # Create [0, 0]
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding
    #print("labels {}".format(i), " ", one_hot_encoding)
    #labels 0   [0, 1]
    #labels 1   [0, 1]
    #labels 2   [0, 1]
    #labels 3   [1, 0]
    #labels 4   [1, 0]
    #labels 5   [1, 0]

#print ("labels {}".format(labels))
# labels [[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]

# Make a word2index dictionary
word2index_map = {}
index =0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
            
#print("word2index_map: ", word2index_map)
# word2index_map:  {'seven': 0, 'one': 1, 'four': 2, 'nine': 3, 'eight': 4, 
# 'pad': 5, 'two': 6, 'five': 7, 'six': 8}
index2word_map = {index: word for word, index in word2index_map.items()}
#print ("index2word_map: ", index2word_map)
# index2word_map:  {0: 'four', 1: 'one', 2: 'seven', 3: 'pad', 
# 4: 'three', 5: 'eight', 6: 'nine', 7: 'six', 8: 'two'}

vocabulary_size = len(index2word_map)
print ("vocabulary_size: ", vocabulary_size)

data_indices = list(range(len(data)))
#print ("data_indices: ", data_indices)

np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
#array(['Six Two Eight Eight Eight PAD', 'Eight Five One Five PAD PAD',
#       'Five Eight Six Eight Six PAD', 'Four Six Eight Four PAD PAD',
#       'Six Two Six Two Four Two', 'Six Five Six Seven Four Four'],
#      dtype='<U29')
labels = np.array(labels)[data_indices]
#print ("labels: ", labels)
#[[0 1]
# [1 0]
# [0 1]
# [1 0]
# [1 0]
# [0 1]]
seqlens = np.array(seqlens)[data_indices]
#print ("seqlens: ", seqlens)
# [5 5 4 5 4 5]

#########################################
# Data split into a train and test
#########################################

train_x = data[:input_num]
print("len of train_x: ", len(train_x))
# len of train_x:  3
train_y = labels[:input_num]
train_seqlens = seqlens[:input_num]

test_x = data[input_num:]
print("len of test_x: ", len(test_x))
# len of train_y:  3

test_y = labels[input_num:]

test_seqlens = seqlens[input_num:]
print("len of test_seqlens: ", len(test_seqlens))

def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x))) # if len(data_x) is 3, instance_indices is [1,2,3]
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    # if batch_size is 2, x is [[1, 0, 4, 0, 4, 3], [4, 7, 0, 0, 0, 3]]
    x = [[word2index_map[word] for word in data_x[i].lower().split()]
        for i in batch]
    y = [data_y[i] for i in batch] # [array([0, 1]), array([1, 0])]
    seqlens = [data_seqlens[i] for i in batch] # [5,5]
    return x,y,seqlens

#########################################
# Create placeholders
#########################################

_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_step])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])


#########################################
# Define embedding
#########################################
with tf.name_scope("embedding"):
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size,
                              embedding_dimension],
                              -1.0, 1.0), name = 'embedding')
    
    print("embeddings shape: ", np.shape(embeddings))  
    # resultEmbeddings shape:  (10, 64)
    print("_inputs shape: ", np.shape(_inputs))        
    # _inputs shape:  (128, 6)    
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
    print("embed shape: ", np.shape(embed))
    # embed shape:  (128, 6, 64)    

#########################################
# Define LSTM layer
#########################################   
""" 
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias = 1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,
                                        sequence_length = _seqlens,
                                        dtype = tf.float32)
"""

#CODE BLOCK FOR MULTIPLE LSTM
num_LSTM_layers = 2
with tf.variable_scope("lstm"):
 
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,
                                             forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell]*num_LSTM_layers,
                                       state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                        sequence_length = _seqlens,
                                        dtype=tf.float32)    
    
    
#########################################
# Define FL layer
#########################################    
    
weights = {
        'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,num_classes],
                                                        mean = 0, stddev = .01))}
biases = {
        'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev = .01))}


#extract the final state and use in a linear layer
final_output = tf.matmul(states[num_LSTM_layers-1][1],
                         weights["linear_layer"]) + biases["linear_layer"]   



#########################################
# Define softmax layer
#########################################    
softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output,
                                                  labels = _labels)
cross_entropy = tf.reduce_mean(softmax)

#########################################
# Define loss function and optimizer
#########################################    
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
cross_prediction = tf.equal(tf.argmax(_labels,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(cross_prediction, tf.float32))) * 100


#########################################
# Run graph
#########################################    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #########################################
    # Training Process
    #########################################    
    
    for step in range(training_steps):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens)
        sess.run(train_step, feed_dict = {_inputs: x_batch, _labels: y_batch,
                                          _seqlens: seqlen_batch})
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict = {_inputs: x_batch,
                                                  _labels: y_batch,
                                                  _seqlens:seqlen_batch})
    
            print("Accuracy at %d: %.5f" % (step, acc))

            

    #########################################
    # Test Process
    #########################################    
            
    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output,1),
                                          accuracy], feed_dict = {_inputs: x_test, _labels: y_test, _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

    output_example = sess.run([outputs], feed_dict = {_inputs:x_test, _labels: y_test, _seqlens: seqlen_test})
    states_1_example = sess.run([states[1]],feed_dict={_inputs:x_test,
                                                   _labels:y_test,
                                                   _seqlens:seqlen_test})

    states_example = sess.run([states],feed_dict={_inputs:x_test,
                                                   _labels:y_test,
                                                   _seqlens:seqlen_test})

        
    print("output_example shape: ", np.shape(output_example))
    print("states_example shape: ", np.shape(states_example))            
    print("states_1_example shape: ", np.shape(states_1_example))        
    print("output_example[0][1][0:6,0:3] ", output_example[0][1][0:6,0:3] )
    print("states_1_example[0][1][1][0:3]] ", states_1_example[0][1][1][0:3])
    print("states_example[0][1][1][1][0:3]] ", states_example[0][1][1][1][0:3])    

    # The last state as state[1] is the same as output example
    
    # output_example shape:  (1, 128, 6, 32)
    # states_1_example shape:  (1, 2, 128, 32)
    # states_example shape:  (1, 2, 2, 128, 32)    
    # states shape:  (1, 2, 128, 32)    
    # output_example[0][1][0:6,0:3]  [[ 0.37918463  0.35131711  0.23618889]
    # [ 0.67244375  0.64754087  0.42223427]
    # [ 0.78063494  0.78508306  0.54675955]
    # [ 0.83856553  0.85074878  0.5793125 ]
    # [ 0.          0.          0.        ]
    # [ 0.          0.          0.        ]]
    # states_1_example[0][1][1][0:3]]  [ 0.83856553  0.85074878  0.5793125 ]
    # states_example[0][1][1][1][0:3]]  [ 0.83856553  0.85074878  0.5793125 ]

    
        



