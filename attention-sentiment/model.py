#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:24:53 2017

@author: gonsoomoon
"""

print("===== model.py ===========")

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.ops import (
        array_ops,
        init_ops,
        variable_scope as vs,
)


from tensorflow.python.ops.math_ops import (
        sigmoid,
        tanh,
)

def _xavier_weight_init (nonlinearity = 'tanh'):
    def _xavier_initializer(shape, **kwargs):
        """
        Tanh and sigmoid initialization
        """
        eps = 1.0 / np.sqrt(np.sum(shape))
        return tf.random_uniform(shape, minval = -eps, maxval = eps)
        
    def _relu_xavier_initializer(shape, **kwargs):
        eps = np.sqrt(2.0) / np.sqrt(np.sum(shape))
        return tf.random_uniform(shape, minval = -eps, maxval = eps)
 
       	
    if nonlinearity in ('tanh', 'sigmoid'):
        return _xavier_initializer
    elif nonlinearity in ('relu'):
        return _relu_xavier_initializer
    else:
        raise Exception('Please choose a valid nonlinearity : tanh | sigmoid | relu')
    
# Linear transformation
def _linear(args, output_size, bias, bias_start=0.0, nonlinearity='relu',scope = None, name = None):
    """
    Sending inputs through a two layer MLP.
    Args:
        args: list of inputs of shape (N, H)
        output_size: second dimension of W
        bias: boolean, whether or not to add bias
        bias_start: initial bias value
        nonlinearity: nonlinear transformation to use (tanh|sigmoid|relu)
        scope: (optional) Variable scope to create parameters in.
        name: (optional) variable name.
    Returns:
        Tensor with shape (N, output_size)
    """

    _input = tf.concat(values = args, axis = 1,)    
    shape = _input.get_shape()
    # Computation
    scope = vs.get_variable_scope()
    
    with vs.variable_scope(scope) as outer_scope:
        w_name = "W_1_"
        if name is not None:
            w_name += name
        W_l = vs.get_variable(
                name = w_name,
                shape = [shape[1], output_size], 
                initializer = _xavier_weight_init(nonlinearity = nonlinearity),)
        result_l = tf.matmul(_input, W_l)
        if bias:
            b_name = "b_l_"
            if name is not None:
                b_name += name
            b_l = vs.get_variable(name = b_name,
                                  shape = (output_size,),
                                  initializer = init_ops.constant_initializer(
                                          bias_start, dtype = tf.float32),)
            result_l = tf.add(result_l, b_l)
        return result_l
    
def ln(inputs, epsilon = 1e-5, scope = None):
    """
    ln(), which is another optimization technique that will normalize our inputs 
    into the GRU (Gated Recurrent Unit) before applying the activation function    
    Compute layer norm given an input tensor.
    We get in an input of shape [N X D] and with LN
    we compute the mean and var for each individual training point
    across all its's hidden dimensions rather than across the training batch
    as we do in BN.
    This gives us a mean and var of shape [N * 1]
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims = True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha', 
                                shape = [inputs.get_shape()[1]],
                                initializer = tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape = [inputs.get_shape()[1]],
                                initializer = tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
    
    return LN

# On the basis of GRU architecture, the layer normalization is added  so that
# the name is custome_GRUCell  
class custom_GRUCell (tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, input_size = None, activation = tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated", self)
        self._num_units = num_units
        self._activation = activation


    def state_size(self):
        return self._num_units

    def output_size(self):
        return self._num_units
    
    def __call___ (self, inputs, state, scope = None):
        """
        Gated recurrent unit (GRU) with nunites cells.
        """
        with vs.variable_scope(scope or type(self).__name__): # GRUCell
            with vs.variable_scope("Gates"): # Reset gate and update gate
                # We start with bias of 1.0 to not reset and not update
                r, u = array_ops.split(
                        _linear([inputs, state]),
                        2 * self._num_units,
                        True,)
                
                # Apply layer normalization to the two gates
                r = ln(r, scope = 'r/')
                u = ln(u, scope = 'u/')
                
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = self._activation(
                        _linear([inputs, r * state],
                                self._num_units, True))
            new_h = u * state + (1-u) * c
        
        return new_h, new_h

# Return the cell wrappered with dropput
class DropoutLayer():    
    def add_dropout_and_layers(self, single_cell, keep_prob, num_layers):
        """
        Add dropout and create stacked layers using a single_cell.
        """
        
        # Dropout
        stacked_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
                                                     output_keep_prob = keep_prob)
        # Each state as one cell
        if num_layers > 1:
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [single_cell] * num_layers)
            
        return stacked_cell
                
"""
Simple GRU Encoder / Decoder Model w/ Attentional Interface
"""
import tensorflow as tf


class Model():
    """
    Tensorflow graph,
    """
    def __init__(self, FLAGS, vocab_size, custom_GRUCell, DropoutLayer):
        self.FLAGS = FLAGS
        self._vsize = vocab_size
        self.custom_GRUCell = custom_GRUCell
        self.DropoutLayer = DropoutLayer
        
    def train(self, sess, batch_reviews, batch_labels,
              batch_review_lens, embeddings, keep_prob):
        """
        Train the model using a batch and predicted guesses.
        """
        outputs = [ self._train_op,
                   self._logits,
                   self._loss,
                   self._accuracy,
                   self._lr,
                   self._z,]
        inputs = {
                self._reviews: batch_reviews,
                self._labels: batch_labels,
                self._review_lens: batch_review_lens,
                self._embeddings: embeddings,
                self._keep_prob: keep_prob,}
        
        return sess.run(outputs, inputs)

    def eval(self, sess, batch_reviews, batch_labels,
             batch_review_lens, embeddings, keep_prob = 1.0):
        """
        Evaluation of validation set
        """
        outputs = [
                self._logits,
                self._loss,
                self._accuracy,]
        inputs = {
                self._reviews: batch_reviews,
                self._labels: batch_labels,
                self._review_lens: batch_review_lens,
                self._embeddings: embeddings,
                self._keep_prob: keep_prob,}
        return sess.run(outputs, inputs)
    
    def infer(self, sess, batch_reviews, batch_labels, batch_review_lens, embeddings,
              keep_prob = 1.0):
        """
        Inference with a sample sentence
        """
        outputs = [
                self._logits,
                self._probabilities,
                #self.loss,
                #self.prediction_labels,
                self._z,]
        inputs = {
                self._reviews: batch_reviews,
                self._labels: batch_labels,                
                self._review_lens: batch_review_lens,
                self._embeddings: embeddings,
                self._keep_prob: keep_prob,}
        return sess.run(outputs, inputs)
    
    def _add_placeholders(self):
        """
        Input that will be fed into our DCN graph.
        """
        print ("--> Adding placeholders:")
        
        FLAGS = self.FLAGS
        self._reviews = tf.placeholder(
                dtype = tf.int32,
                shape = [None, FLAGS.max_input_length],
                name = 'reviews')
        self._review_lens = tf.placeholder(
                dtype = tf.int32,
                shape = [None,],
                name = "review_lens")
        self._labels = tf.placeholder(
                dtype = tf.int32,
                shape = [None,],
                name = "labels")
        self._embeddings = tf.placeholder(
                dtype = tf.float32,
                shape = [FLAGS.vocab_size, FLAGS.emb_size],
                name = 'glove_embeddings')
        self._keep_prob = tf.placeholder(
                dtype = tf.float32,
                shape = (),
                name = "keep_prob")
        
        print ("\t self._reviews:", self._reviews.get_shape())
        print ("\t self._labels:", self._labels.get_shape())
        print ("\t self._embeddings:", self._embeddings.get_shape())
        print ("\t self._keep_prob:", self._keep_prob.get_shape())

            
        # --> Adding placeholders:
        #	 self._reviews: (?, 300)
        #	 self._labels: (?,)
        #	 self._embeddings: (75133, 200)
        #	 self._keep_prob: ()
        
       
    def _build_encoder(self):
        """
        Constructing the encoder
        """
        print ("==> Building the encoder:")
        
        FLAGS = self.FLAGS
        batch_size = FLAGS.batch_size
        hidden_size = FLAGS.hidden_size
        max_input_length = FLAGS.max_input_length
        
        print("batch_size: ", batch_size)
        print("hidden_size: ", hidden_size)
        print("max_input_length: ", max_input_length)        
        # batch_size:  256
        # hidden_size:  200
        # max_input_length:  300
                
        with tf.variable_scope('embeddings'):
            print("\t embeddings:")
            
            if FLAGS.embedding == 'random':
                # Random embedding weights
                embedding = tf.get_variable(
                        name = 'embeddings',
                        shape = [self._vsize, FLAGS.emb_size],
                        dtype = tf.float32,
                        initializer = tf.truncated_normal_initializer(stddev=1e-4),
                        trainable = FLAGS.train_embedding)
            elif FLAGS.embedding == 'glove':
                # Glove embedding weights
                embedding = self._embeddings

                
            # Check embedding dim
            if embedding.get_shape()[1] != FLAGS.emb_size:
                raise Exception(
                        "Embedding's dimension does not match specified emb_size.")
                
            # Embedding the review
            
            fn = lambda x: tf.nn.embedding_lookup(embedding, x)
            c_embedding = tf.map_fn(
                    lambda x: fn(x), self._reviews, dtype = tf.float32)
            
            # The following statement is the same as the above statements
            # c_embedding = tf.nn.embedding_lookup(embedding, self._reviews)
            
            print("\t\t embedding:", embedding.get_shape())
            print("\t\t reviews_embedded: ", c_embedding.get_shape())
            # embeddings:
            # embedding: (75133, 200)
            # reviews_embedded:  (?, 300, 200)
            
   
        
        with tf.variable_scope('c_encoding'):
            print ("\t c_encoding:")
            
            # GRU cells
            drop_out_layer = self.DropoutLayer()
            #custom_gru_cell = self.custom_GRUCell(hidden_size)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias = 1.0)
            
            cell = drop_out_layer.add_dropout_and_layers(
                    single_cell = lstm_cell,
                    #single_cell = custom_gru_cell(),                    
                    keep_prob = self._keep_prob,
                    num_layers = FLAGS.num_layers,)
            # Dynamic-GRU
            # return (outputs, last_output_states (relevant))
            all_outputs, h = tf.nn.dynamic_rnn(
                    cell = cell,
                    inputs = c_embedding,
                    #sequence_length = self._review_lens,
                    dtype = tf.float32,)
                    #time_major = False,)
            
            self._all_outputs = all_outputs
            self._h = h
            self._z = all_outputs
            
            print("\t\t self._all_outputs", self._all_outputs.get_shape())
            
            #self._all_outputs (?, 300, 200)
            #print("\t\t self._h", self._h.get_shape())
            #print("\t\t self._h", self._h.shape)
        
            
    def _build_attentional_interface(self):
        """
        Adding an attentional interface
        for model interpretability.
        """
        print ("==> Building the attentional interface:")
        FLAGS = self.FLAGS
        batch_size = FLAGS.batch_size
        hidden_size = FLAGS.hidden_size
        #max_input_length = FLAGS.max_input_length
        loop_until = tf.to_int32(np.array(range(batch_size)))
        
        
        with tf.variable_scope('attention') as attn_scope:
            print ("\t attention:")
            
            # Time-major self._all_outputs (N, M, H) --> (M, N, H)
            all_outputs_time_major = tf.transpose(self._all_outputs, perm=[1,0,2])
            
            # Apply tanh nonlinearity
            fn = lambda _input: tf.nn.tanh(_linear(
                    args = _input,
                    output_size = hidden_size,
                    bias = True,
                    bias_start = 0.0,
                    nonlinearity = 'tanh',
                    scope = attn_scope,
                    name = 'attn_nonlinearity',))
            
            z = tf.map_fn(lambda x: fn(x), all_outputs_time_major, dtype=tf.float32)
            
            # Apply softmax weights
            fn = lambda _input: tf.nn.tanh(_linear(
                    args = _input,
                    output_size =1,
                    bias = True,
                    bias_start = 0.0,
                    nonlinearity = 'tanh',
                    scope = attn_scope,
                    name = 'attn_softmax',))
            
            z = tf.map_fn(
                    lambda x: fn(x), z, dtype= tf.float32)
            
            # Squeeze and convert to batch major
            z = tf.transpose(
                    tf.squeeze(
                            input = z,
                            axis = 2,),
                    perm = [1,0])
                    
            # Normalize
            self._z = tf.nn.softmax(
                    logits = z,)


            
            # Create context vector (via soft attention.)
            fn = lambda sample_num: tf.reduce_sum(
                    tf.multiply(
                                self._all_outputs[sample_num][:self._review_lens[sample_num]],
                                # (500,) --> (500, 1) --> (500, 200)
                                tf.tile(
                                        input = tf.expand_dims(
                                                self._z[sample_num][:self._review_lens[sample_num]],1),
                                        multiples = (1, hidden_size),
                                )),
                        axis = 0)
                                        
            self._c = tf.map_fn(
                    lambda sample_num: fn(sample_num), loop_until, dtype = tf.float32 )
            print("\t\t self._Z", self._z.get_shape())
            print("\t\t self._c", self._c.get_shape())
            # attention:
            #		 Alpha vector: self._Z (?, 300)
            #		 Context Vector: self._c (256, 200)            
            
            
    def _build_decoder(self):
        """
        Applying a softmax on output of encoder
        """
        print("==> Building the decoder:")
        self._logits = _linear(
                args = self._c, # self._c (with attn) or self._h (no attn)
                output_size = self.FLAGS.num_classes,
                bias = True,
                bias_start =0.0,
                nonlinearity = 'relu',
                name = 'softmax_op',                
                )
        self._probabilities = tf.nn.softmax(
                logits = self._logits,)
        print ("\t\t self._logits ", self._logits.get_shape())
        print ("\t\t self._probabilities", self._probabilities.get_shape())                            
        # Building the decoder:
	    # self._logits  (256, 2)
        # self._probabilities (256, 2)
        
        
    def _add_loss(self):
        """
        Determine the loss
        """
        print ("==> Establishing the loss function.")
        self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = self._labels, 
                        logits = self._logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._logits,1),
                                                        tf.cast(self._labels, tf.int64)), tf.float32))
        self.prediction_labels = tf.argmax(self._logits,1)
        print("self.loss: " , self.loss.shape)
        print("Type of the self.loss: " , type(self.loss))        
        
        return self.loss, self.accuracy, self.prediction_labels
    
    def _add_train_op(self):
        """
        Add the training optimizer
        """
        print("==> Creating the training optimizer.")
        
        # Decay learning rate
        self._lr = tf.maximum(
                self.FLAGS.min_lr,
                tf.train.exponential_decay(
                        learning_rate = self.FLAGS.lr,
                        global_step = self.global_step,
                        decay_steps = 100000,
                        decay_rate = self.FLAGS.decay_rate,
                        staircase = False,))
        
        # Initialize the optimizer
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate = self._lr).minimize(self.loss,
                                        global_step = self.global_step)
        
        return self.optimizer
    
    def _build_graph(self):
        """
        Construct each component
        """
        self._add_placeholders()
        
        self._build_encoder()
        
        self._build_attentional_interface()
        
        self._build_decoder()
        

        self.global_step = tf.Variable(0, trainable = False)  
        if self.FLAGS.mode == 'train':
            print("Mode: Train")
            self._loss, self._accuracy, self._prediction_labels = self._add_loss()
            self._train_op = self._add_train_op()
       
        # Components for model saving
        self.saver = tf.train.Saver(tf.global_variables())
        print ("==> Review Classifier built!")
    
        
def generate_epoch(data_path, num_epochs, batch_size):
    """
    Generate num_epoch epochs.
    Args:
        data_path: path for train.p|valid.p
        num_epochs: number of epochs to run for
        batch_size: samples per each batch
    """
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)
        
    processed_contexts, processed_answers = [], []
    context_lens = []
    
    for entry in entries:
        processed_contexts.append(entry[0])
        context_lens.append(entry[1])
        processed_answers.append(entry[2])
    
    features = [processed_contexts, processed_answers]
    print("featurs len: ", len(features[0]))
    #print("features: ", features)
    seq_lens = [context_lens,]
    print("seq_lens len: ", len(seq_lens))
    #print("seq_lens: ", seq_lens)    
    
    for epoch_num in range(num_epochs):
        yield generate_batch(features, seq_lens, batch_size)
        
def generate_batch(features, seq_lens, batch_size):
        
    """
    Generate batches of size <batch_size>
    Args:
        features: processed contexts, questions and answers.
        seq_lens: context and question actual(pre-padding) seq-lens.
        batch_size: samples per each batch
    """
    data_size = len(features[0])
    print("data_size: ", data_size)
    num_batches = data_size // batch_size
    print("num_batches: ", num_batches)
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        
        batch_features = []
        for feature in features:
            batch_features.append(feature[start_index:end_index])
        batch_lens = []
        for seq_len in seq_lens:
            batch_lens.append(seq_len[start_index:end_index])
            
        yield batch_features, batch_lens
        
class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self, num_epoch, model_name):
        """
        """
        self.data_dir = "data/processed_reviews" # location of reviews data
        self.ckpt_dir = "data/processed_reviews/ckpt" # locations of model checkpoints
        self.mode = "train" # train | infer
        self.model = "new" # odl | new
        self.lr = 1e-4 # learning rate
        self.num_epochs = num_epoch # num of epochs
        self.batch_size = 256 # batch_size
        self.hidden_size = 200 # num hidden units for RNN
        self.embedding = "glove" # random | glove
        self.emb_size = 200 # num hidden units for embeddings
        self.max_grad_norm = 5 # max gradient norm
        self.keep_prob = 0.9 # Keep prob for dropout layers
        self.num_layers = 1 # number of layers for recurrsion
        self.max_input_length = 300 # max number of words per review
        #self.min_lr = 1e-6 # minimum learning rate
        self.min_lr = 1e-1 # minimum learning rate        
        self.decay_rate = 0.96 # Decay rate for lr per global step (train batch)
        self.save_every = 10 # Save the model every <save_every> epochs
        self.model_name = model_name
        
def create_model(sess, FLAGS, vocab_size, custom_GRUCell, DropoutLayer):
    """
    Creates a new model or loads old one.
    """
    imdb_model = Model(FLAGS, vocab_size, custom_GRUCell, DropoutLayer)
    imdb_model._build_graph()
    
    print("FLAGS.model: ", FLAGS.model)
    
    if FLAGS.model == 'new':
        print ('==> Create a new model.')
        sess.run(tf.global_variables_initializer())
    elif FLAGS.model == 'old':
        ckpt_path = os.path.join(FLAGS.basedir, FLAGS.ckpt_dir, FLAGS.model_name) 
        print("ckpt_path: ", ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print("In create_model (), ckpt: ", ckpt)
        print("In create_model (), ckpt.model_checkpoint_path: ", ckpt.model_checkpoint_path)        
        if ckpt and ckpt.model_checkpoint_path:
            print("==> Restoring old model parameters from %s" %
                  ckpt.model_checkpoint_path)
            imdb_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("==> No old model to load from sor initializing a new one.")
            sess.run(tf.global_variables_initializer())

                
    return imdb_model
        
from vocabulary import Vocab
import pickle

def train(FLAGS, basedir, custom_GRUCell, DropoutLayer ):
    """
    Train a previous or new model.
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/processed_reviews/vocab.txt')
    train_data_path = os.path.join(
        basedir, 'data/processed_reviews/train.p')
    validation_data_path = os.path.join(
        basedir, 'data/processed_reviews/validation.p')
    vocab = Vocab(vocab_path)
    FLAGS.num_classes = 2

    # Load embeddings (if using GloVe)
    if FLAGS.embedding == 'glove':
        with open(os.path.join(
            basedir, 'data/processed_reviews/embeddings.p'), 'rb') as f:
            embeddings = pickle.load(f)
        FLAGS.vocab_size = len(embeddings)

    # Start tensorflow session
    with tf.Session() as sess:

        # Create|reload model
        imdb_model = create_model(sess, FLAGS, len(vocab), custom_GRUCell, DropoutLayer )
   
        

    
        # Metrics
        metrics = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
        }

        # Store attention score history for few samples
        attn_history = {
            "sample_0":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_1":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_2":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_3":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_4":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
        }

        # Start training
        for train_epoch_num, train_epoch in enumerate(generate_epoch(
                train_data_path, FLAGS.num_epochs, FLAGS.batch_size)):

            print ("==> EPOCH:", train_epoch_num)

            for train_batch_num, (batch_features, batch_seq_lens) in \
                enumerate(train_epoch):

                batch_reviews, batch_labels = batch_features
                batch_review_lens, = batch_seq_lens

                # Display shapes once
                if (train_epoch_num == 0 and train_batch_num == 0):
                    print ("Reviews: ", np.shape(batch_reviews))
                    print ("Labels: ", np.shape(batch_labels))
                    print ("Review lens: ", np.shape(batch_review_lens))
                    # ==> EPOCH: 0
                    # Reviews:  (256, 300)
                    # Labels:  (256,)
                    # Review lens:  (256,)

                _, train_logits, train_loss, train_acc, lr, attn_scores = \
                    imdb_model.train(
                        sess=sess,
                        batch_reviews=batch_reviews,
                        batch_labels=batch_labels,
                        batch_review_lens=batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=FLAGS.keep_prob,
                        )
            print("train_batch_num: ", train_batch_num)


            for valid_epoch_num, valid_epoch in \
                enumerate(generate_epoch(
                    data_path=validation_data_path,
                    num_epochs=1,
                    batch_size=FLAGS.batch_size,
                    )):

                for valid_batch_num, (valid_batch_features, valid_batch_seq_lens) in \
                    enumerate(valid_epoch):

                    valid_batch_reviews, valid_batch_labels = valid_batch_features
                    valid_batch_review_lens, = valid_batch_seq_lens

                    valid_logits, valid_loss, valid_acc = imdb_model.eval(
                        sess=sess,
                        batch_reviews=valid_batch_reviews,
                        batch_labels=valid_batch_labels,
                        batch_review_lens=valid_batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=1.0, # no dropout for val|test
                        )

            print("valid_epoch_num: ", valid_epoch_num)
            print ("[EPOCH]: %i, [LR]: %.6e, [TRAIN ACC]: %.3f, [VALID ACC]: %.3f " \
                   "[TRAIN LOSS]: %.6f, [VALID LOSS]: %.6f" % (
                train_epoch_num, lr, train_acc, valid_acc, train_loss, valid_loss))
            # [EPOCH]: 0, [LR]: 9.999918e-05, [TRAIN ACC]: 0.543, 
            # [VALID ACC]: 0.480 [TRAIN LOSS]: 0.683403, [VALID LOSS]: 0.692266
            # ==> Saving the model.


            # Store the metrics
            metrics["train_loss"].append(train_loss)
            metrics["valid_loss"].append(valid_loss)
            metrics["train_acc"].append(train_acc)
            metrics["valid_acc"].append(valid_acc)

            # Store attn history
            for i in range(5):
                sample = "sample_%i"%i
                attn_history[sample]["review"] = batch_reviews[i]
                attn_history[sample]["label"] = batch_labels[i]
                attn_history[sample]["review_len"] = batch_review_lens[i]
                attn_history[sample]["attn_scores"].append(attn_scores[i])

            # Save the model (maybe)
            if ((train_epoch_num == (FLAGS.num_epochs-1)) or
            ((train_epoch_num%FLAGS.save_every == 0) and (train_epoch_num>0))):

                # Make parents ckpt di231r if it does not exist
                if not os.path.isdir(os.path.join(basedir, FLAGS.data_dir, 'ckpt')):
                    os.makedirs(os.path.join(basedir, FLAGS.data_dir, 'ckpt'))

                # Make child ckpt dir for this specific model
                if not os.path.isdir(os.path.join(basedir, FLAGS.ckpt_dir)):
                    os.makedirs(os.path.join(basedir, FLAGS.ckpt_dir))

                checkpoint_path = \
                    os.path.join(
                        basedir, FLAGS.ckpt_dir,FLAGS.model_name, "%s.ckpt" % FLAGS.model_name)

                print ("==> Saving the model.")
                imdb_model.saver.save(sess, checkpoint_path,
                                 global_step=imdb_model.global_step)
                print("checkpoint_path: ", checkpoint_path)

    """
    """

    # Save the metrics
    metrics_file = os.path.join(basedir, FLAGS.ckpt_dir,FLAGS.model_name, 'metrics.p')
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    print("metrics_file: ", metrics_file)

    # Save the attention scores
    attn_history_file = os.path.join(basedir, FLAGS.ckpt_dir,FLAGS.model_name, 'attn_history.p')
    with open(attn_history_file, 'wb') as f:
        pickle.dump(attn_history, f)
    print("attn_history_file: ", attn_history_file)
    
