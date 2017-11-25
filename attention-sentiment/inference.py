#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:46:40 2017

@author: gonsoomoon
"""
import model 
from vocabulary import Vocab

class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """
        self.basedir = ''
        self.data_dir = "data/processed_reviews" # location of reviews data
        self.ckpt_dir = "data/processed_reviews/ckpt/" # locations of model checkpoints
        self.mode = "infer" # train | infer
        self.model = "old" # odl | new
        self.lr = 1e-4 # learning rate
        self.num_epochs = 1 # num of epochs
        self.batch_size = 256 # batch_size
        self.hidden_size = 200 # num hidden units for RNN
        self.embedding = "glove" # random | glove
        self.emb_size = 200 # num hidden units for embeddings
        self.max_grad_norm = 5 # max gradient norm
        self.keep_prob = 0.9 # Keep prob for dropout layers
        self.num_layers = 1 # number of layers for recurrsion
        self.max_input_length = 300 # max number of words per review
        self.min_lr = 1e-6 # minimum learning rate
        self.decay_rate = 0.96 # Decay rate for lr per global step (train batch)
        self.save_every = 10 # Save the model every <save_every> epochs
        self.model_name = "epoch20" 

FLAGS = parameters()

import os, pickle
import tensorflow as tf
basedir = ''

def inference():
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/processed_reviews/vocab.txt')
    validation_data_path = os.path.join(
        basedir, 'data/processed_reviews/validation.p')
    vocab = Vocab(vocab_path)
    FLAGS.num_classes = 2
    
    # Store attention score history for few samples
    attn_history = {
        "sample_0":
        {"review": None, "pred_label": None,"label": None, "review_len": None, "attn_scores": []},
        "sample_1":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_2":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_3":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_4":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_5":
        {"review": None, "pred_label": None,"label": None, "review_len": None, "attn_scores": []},
        "sample_6":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_7":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_8":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        "sample_9":
        {"review": None, "pred_label": None, "label": None, "review_len": None, "attn_scores": []},
        
    }
    

    # Load embeddings (if using GloVe)
    if FLAGS.embedding == 'glove':
        with open(os.path.join(
            basedir, 'data/processed_reviews/embeddings.p'), 'rb') as f:
            embeddings = pickle.load(f)
        FLAGS.vocab_size = len(embeddings)

    # Start tensorflow session
    with tf.Session() as sess:
        # Create|reload model
        imdb_model = model.create_model(sess, FLAGS, len(vocab), model.custom_GRUCell, model.DropoutLayer )
        
        for valid_epoch_num, valid_epoch in \
            enumerate(model.generate_epoch(
                data_path=validation_data_path,
                num_epochs=1,
                batch_size=FLAGS.batch_size,
                )):

            for valid_batch_num, (valid_batch_features, valid_batch_seq_lens) in \
                enumerate(valid_epoch):

                valid_batch_reviews, valid_batch_labels = valid_batch_features
                valid_batch_review_lens, = valid_batch_seq_lens

                """
                valid_logits, valid_loss, valid_acc = imdb_model.eval(
                    sess=sess,
                    batch_reviews=valid_batch_reviews,
                    batch_labels=valid_batch_labels,
                    batch_review_lens=valid_batch_review_lens,
                    embeddings=embeddings,
                    keep_prob=1.0, # no dropout for val|test
                    )
                """
                valid_logits, valid_prob, valid_z = imdb_model.infer(
                    sess=sess,
                    batch_reviews=valid_batch_reviews,
                    batch_labels=valid_batch_labels,
                    batch_review_lens=valid_batch_review_lens,
                    embeddings=embeddings,
                    keep_prob=1.0, # no dropout for val|test
                    )

                pred_labels = sess.run(tf.argmax(valid_logits,1))
                
                
                print("valid_epoch_num: ", valid_epoch_num)
                print("valid_batch_num: ", valid_batch_num)
                print("valid_batch_reviews: ", len(valid_batch_reviews))
                
        # Store attn history
        for i in range(10):
            sample = "sample_%i"%i
            print("prob: ", valid_prob[i])
            print("pred_labels[i] : ", pred_labels[i])
            print("valid_batch_labels[i]: ", valid_batch_labels[i])
            attn_history[sample]["review"] = valid_batch_reviews[i]
            attn_history[sample]["pred_label"] = pred_labels[i]            
            attn_history[sample]["label"] = valid_batch_labels[i]
            attn_history[sample]["review_len"] = valid_batch_review_lens[i]
            attn_history[sample]["attn_scores"].append(valid_z[i])
          
        # Save the attention scores
        attn_history_file = os.path.join(basedir, FLAGS.ckpt_dir,FLAGS.model_name, 'attn_history_epoch20.p')
        with open(attn_history_file, 'wb') as f:
            pickle.dump(attn_history, f)
        print("attn_history_file: ", attn_history_file)
            
                
        #print("valid_epoch_num: ", valid_epoch_num)
        #print("valid_prob shape: ", valid_prob.shape)
        #print("valid_logits: ", valid_logits)
        #print("valid_prob: ", valid_prob)        
        """
        print ("[EPOCH]: %i, [VALID logits]: %.3f " \
               "[VALID prob]: %.6f" % (
            valid_logits, valid_prob))
        """
