#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:49:10 2017

@author: gonsoomoon
"""

import input_word_data as data
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

# Hyperparameters
batch_size = 32
embedding_size = 128
skip_window = 5
num_skips = 4
batches_per_epoch = int(data.data_size * num_skips / batch_size)
# 2125650 = 17005207 * 4 / 32
training_epochs = 5
neg_size = 64
display_step = 2000
val_step = 100000
learning_rate = 0.1

print ("Epochs: %d, Batches per epoch: %d, Examples per batch: %d" % (training_epochs, batches_per_epoch, batch_size))

val_size = 5
val_dist_span = 500
val_examples = np.random.choice(val_dist_span, val_size, replace = False)
top_match = 8
plot_only = 500

# Encoder producing embedding, which is a matrix of vocabuary size of x * embedding size
# embedding_shape = vocabuary size * embedding size
def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable("E", initializer = embedding_init)
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix

# Decoder producing the loss function
# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels 
# each time we evaluate the loss

def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape,y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev = 1.0/(weight_shape[1]**0.5))
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer = nce_weight_init)
        nce_b = tf.get_variable("b", initializer = nce_bias_init)
        
        total_loss = tf.nn.nce_loss(nce_W, nce_b,
                                    y, embedding_lookup,
                                    neg_size, data.vocabulary_size )
        return tf.reduce_mean(total_loss)

# Training Operation    
def training(cost, global_step):
    with tf.variable_scope("training"):
        summary_op = tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step = global_step)
        return train_op, summary_op
    
# Compute the cosine similarity between minibatch examples and all embeddings    
def validation(embedding_matrix, x_val):
    norm = tf.reduce_sum(embedding_matrix ** 2, 1, keep_dims = True) ** 0.5
    normalized = embedding_matrix / norm
    val_embeddings = tf.nn.embedding_lookup(normalized, x_val)
    cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b = True)
    return normalized, cosine_similarity

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.variable_scope("skipping_model"):
            x = tf.placeholder(tf.int32, shape = [batch_size])
            y = tf.placeholder(tf.int32, [batch_size, 1])
            val = tf.constant(val_examples, dtype = tf.int32)
            global_step = tf.Variable(0, name = 'global_step', trainable=False)
            
            # For the batch_size = 32, embedding_size = 128            
            # data.vocabulary_size:  1000  embedding_size :  128
            # e_lookup:  (32, 128)  e_matrix :  (1000, 128)            
            e_lookup, e_matrix = embedding_layer(x, [data.vocabulary_size, embedding_size])
            print ('data.vocabulary_size: ', data.vocabulary_size, ' embedding_size : ', embedding_size)
            # Cost function: a mean of the NCE cost
            # Input:
            #   e_lookup: input
            #   [data.vocabulary_size, embedding_size]: weight with the e_lookup's shape 
            #   [data.vocabulary_size]: bias with length of vocabulary
            #   y: label
            # Output:
            #   Cost: Take the sum of the probabilities corresponding to the noncontext comparisons
            #         and subtract the probability corresponding to the context comparison
            cost = noise_contrastive_loss(e_lookup,
                                          [data.vocabulary_size, embedding_size],
                                          [data.vocabulary_size],y)
            train_op, summary_op = training(cost, global_step)
            # Validation Op
            val_op = validation(e_matrix, val)
            sess = tf.Session()
            train_writer = tf.summary.FileWriter("skipgram_logs/", graph = sess.graph)
            sess.run(tf.global_variables_initializer())
            step = 0
            avg_cost = 0
            
            for epoch in range(training_epochs):
                for minibatch in range(batches_per_epoch):
                    step +=1
                    minibatch_x, minibatch_y = data.generate_batch(batch_size, num_skips, skip_window)
                    feed_dict = {x: minibatch_x, y: minibatch_y}
                    _, new_cost, train_summary = sess.run([train_op, cost, summary_op], feed_dict = feed_dict)
                    #_, new_cost, train_summary, r_e_lookup, r_e_matrix = sess.run([train_op, cost, summary_op, e_lookup, e_matrix], feed_dict = feed_dict)
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    
                    avg_cost += new_cost / display_step
                    
                    if step % display_step == 0:
                        print ("Elapsed:", str(step), "batches, Cost = ", "{:.9f}".format(avg_cost))
                        #print('r_e_lookup: ' , np.shape(r_e_lookup), ' r_e_matrix : ', np.shape(r_e_matrix))
                        #print(' r_e_matrix : ', r_e_matrix)                        
                        # r_e_lookup:  (32, 128)  r_e_matrix :  (1000, 128)
                        
                        avg_cost = 0
                        
                    
                    # Print out the nearest 8 words based on similarity
                    if step % val_step == 0:
                        _, similarity = sess.run(val_op)
                        for i in range(val_size):
                            val_word = data.reverse_dictionary[val_examples[i]]
                            neighbors = (-similarity[i,:]).argsort()[1:top_match + 1]
                            print_str = "Nearest neighbor of %s:" % val_word
                            for k in range(top_match):
                                print_str += " %s," % data.reverse_dictionary[neighbors[k]]
                            print (print_str[:-1])
                    
                            
            final_embeddings, _ = sess.run(val_op)
            print('final_embedding: ' , np.shape(final_embeddings))
                    
            
    # t-distributed stochastic neighbor embedding            
    # Redice 128 of embeddings to 2 of embeddings per word that is x,y coordinate
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_embeddings = np.asfarray(final_embeddings[:plot_only,:], dtype='float')
    print('plot_embedding: ' , np.shape(plot_embeddings))
    # low_dim_embs: 2 embedding as (x,y) coordinate
    low_dim_embs = tsne.fit_transform(plot_embeddings)
    print('low_dim_embs: ' , np.shape(low_dim_embs))    
    # label as word
    labels = [data.reverse_dictionary[i] for i in range(plot_only)]
    print('labels: ' , np.shape(labels))        
    
    data.plot_with_labels(low_dim_embs, labels)
                        
            
    