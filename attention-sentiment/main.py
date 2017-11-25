#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:51:17 2017

@author: gonsoomoon
"""

basedir = ""

import os
import vocabulary as voc
import model as model


print("===== main.py ===========")

############################
# Test Vocabulary class
############################
"""
FLAGS = voc.parameters()
voc.sample_data(FLAGS.data_dir)
"""

############################
# Train Model class
############################
"""
FLAGS = model.parameters(1, "epoch01")
model.train(FLAGS, basedir, model.custom_GRUCell, model.DropoutLayer)
"""


############################
# Show metrics on a chart
############################


import plot as plot

FLAGS = plot.parameters("epoch20")
FLAGS.ckpt_dir = FLAGS.ckpt_dir + '/%s' % (FLAGS.model_name)
plot.plot_metrics(FLAGS, basedir)


############################
# Infer using old model
############################
"""
import inference as infer
infer.inference()
"""

############################
# Draw attention history
############################
"""
import attn_history as attn

FLAGS = attn.parameters("epoch20", 9)
#print("FLAGS.ckpt_dir: ", FLAGS.data_dir)
FLAGS.ckpt_dir = FLAGS.ckpt_dir + '/%s' % (FLAGS.model_name)
attn.process_sample_attn(FLAGS)
"""



