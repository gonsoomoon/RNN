#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:22:52 2017

@author: gonsoomoon
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

#######################################################
# Hyperparameters
#######################################################
input_num = 10000 # number of sentences to be generated 
batch_size = 32
step_num = 1000 # number of training steps
embedding_dimension = 5
negative_samples = 8
LOG_DIR = "logs/word2vec_intro"


#######################################################
# Generating Input
#######################################################
digit_to_word_map = {1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five",
                     6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

sentences = []

# Create two kinds of sentences - sequences of odd and even digits.
for i in range (input_num):
    rand_odd_ints = np.random.choice(range(1,10,2),3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2,10,2),3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    
#print("sentences: ", sentences)
#['One Three Seven', 'Six Four Eight', 'One Three Three', 
#'Two Six Four', 'Three Nine Seven', 'Four Six Four']
 
word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
index2word_map = {index : word for word, index in word2index_map.items()}
print("index2word_map: ", index2word_map)
# index2word_map:  {0: 'five', 1: 'seven', 2: 'three', 3: 'six', 4: 'eight', 5: 'one', 6: 'nine', 7: 'four', 8: 'two'}
vocabulary_size = len(index2word_map)
print("voc size : ", vocabulary_size)
# voc size :  9

#######################################################
# Generating Skip gram as an input
#######################################################
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    #print ("tokenized_sent: ", tokenized_sent)
    # tokenized_sent:  ['five', 'seven', 'three']
    # tokenized_sent:  ['six', 'six', 'eight']
    for i in range(1, len(tokenized_sent) -1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                              word2index_map[tokenized_sent[i]]]
        #print("word_context_pair: ", word_context_pair)
        # Make [Context, Target]
        # word_context_pair:  [[0, 2], 1]
        # word_context_pair:  [[3, 4], 3]
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][1]])    
        #print ("skip_gram_pairs: ", skip_gram_pairs)    
        # skip_gram_pairs:  [[1, 0], [1, 2]]
        # skip_gram_pairs:  [[1, 0], [1, 2], [3, 3], [3, 4]]   
     
#######################################################
# Making a batch of input consisting of x and y
#######################################################        
def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    
    # target as x, context as y
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    
    #print ('batch: ', batch)
    #print ('x_batch: ', x)
    #print('y_batch: ', y)
    # batch:  [242, 147, 263, 149, 341, 376, 236, 142, 275, 207, 227, 73, 27, 1, 315, 339, 103, 369, 394, 157, 279, 161, 120, 81, 313, 302, 387, 218, 206, 253, 194, 176]
    # x_batch:  [5, 5, 5, 0, 8, 7, 1, 2, 3, 6, 5, 4, 3, 1, 2, 6, 6, 1, 2, 0, 5, 4, 7, 7, 4, 5, 3, 3, 6, 8, 5, 0]
    # y_batch:  [[5], [2], [6], [1], [4], [0], [7], [2], [2], [6], [5], [0], [6], [0], [6], [2], [2], [7], [3], [7], [2], [8], [4], [1], [0], [5], [3], [2], [6], [0], [2], [0]]    
    
    return x,y

#x_batch, y_batch = get_skipgram_batch(batch_size)
#[index2word_map[word] for word in x_batch] 
#[index2word_map[word[0]] for word in y_batch]

#######################################################
# Define placeholdres
#######################################################
train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

#######################################################
# Define embedding
# Graph: train_inputs --> embed
#######################################################
# Define embedding
with tf.name_scope("embedding"):
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_dimension],
                              -1.0, 1.0), name = 'embedding')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#######################################################
# NCE Loss function
# Graph: train_inputs --> embed --> loss
#######################################################
# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_dimension],
                            stddev = 1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(
        tf.nn.nce_loss(weights = nce_weights, biases = nce_biases,
                       inputs = embed, labels = train_labels,
                       num_sampled = negative_samples,
                       num_classes = vocabulary_size))
tf.summary.scalar("NCE_LOSS", loss)

#######################################################
# Learning rate decay
#######################################################
global_step = tf.Variable(0, trainable = False)
learningRate = tf.train.exponential_decay(learning_rate = 0.1,
                                          global_step = global_step,
                                          decay_steps = 1000,
                                          decay_rate = 0.95,
                                          staircase = True)

#######################################################
# Define optimizer
# Graph: train_inputs --> embed --> loss, learningRate --> train_step
#######################################################
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
merged = tf.summary.merge_all()

#######################################################
# Run graph
#######################################################
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR, graph = tf.get_default_graph())
    saver = tf.train.Saver()
    
    #######################################################
    # Setting up an embedding configuration in tensorboard
    #######################################################
    # Make metafile for visualization on the embedding tab in tensorboard    
    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))
            # Name	Class
            # one	     0
            # seven	1
            # two	     2
            # four	3
            # three	4
            # six	    5
            # eight	6
            # five	7
            # nine	8
    
    # Create config object    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Attach the embeddings to the config object
    embedding.tensor_name = embeddings.name
    
    # Link this tensor to its metadata file (e.g. labels)
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    print("embedding's metadata_path: ", embedding.metadata_path)    
    # embedding's metadata_path:  logs/word2vec_intro/metadata.tsv
    
    projector.visualize_embeddings(train_writer, config)
    
    tf.global_variables_initializer().run()
    
    #######################################################
    # Training the graph
    #######################################################

    for step in range(step_num):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        # Train train_step
        summary, _ = sess.run([merged, train_step],
                              feed_dict = {train_inputs: x_batch,
                                           train_labels: y_batch})
        train_writer.add_summary(summary, step)
        
        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            # Calculate the loss
            loss_value = sess.run(loss, feed_dict = {train_inputs: x_batch,
                                                     train_labels: y_batch})
            print(" Loss at %d: %.5f" % (step, loss_value))
            
            # r_embeddings = sess.run(embeddings,feed_dict = {train_inputs: x_batch,train_labels: y_batch}) 
            #r_learningRate = sess.run(learningRate,feed_dict = {global_step: step}) 
            #print('learningRate: ', r_learningRate)
            # learningRate:  0.1


    #######################################################
    # Normalize embeddings before using
    #######################################################
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
    # print("embeddings shape: ", np.shape(r_embeddings))
    # print("norm shape: ", np.shape(norm)) 
    # embeddings shape:  (9, 5)
    # norm shape:  (9, 1)

    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)
    print("normalized_embeddings_matrix shape: ", np.shape(normalized_embeddings_matrix))
    # normalized_embeddings_matrix shape:  (9, 5)        
    
ref_word = normalized_embeddings_matrix[word2index_map["one"]]
print("ref_word: ", ref_word)
print("ref_word: ", np.shape(ref_word))
# ref_word:  [-0.45553312 -0.0311466  -0.18242472  0.84984607 -0.18974335]
# ref_word:  (5,)
cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
print("cosine_dists' type: ", type(cosine_dists))
print("cosine_dists: ", cosine_dists)
print("cosine_dists: ", np.shape(cosine_dists))

# cosine_dists:  [ 0.48217091 -0.1437946   0.3461726   1.00000024  0.4710829   0.22736092
#  -0.02905574  0.35576737  0.09994133]
# cosine_dists:  (9,)

# Sort in a decending with -1 and select a range from 1 to 10
ff = np.argsort(cosine_dists)[::-1][1:10]
print("ff: ", ff)
# ff:  [0 4 7 2 5 8 6 1]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])
    
    
    





    
    
    




    