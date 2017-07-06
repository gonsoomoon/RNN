#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:35:38 2017
@author: gonsoomoon
"""
################################
# File: count1sBinaryString.py
# Execution time: 48 minutes on mac pro i5, 16GB
# Tensorflow version: 1.0
################################

import numpy as np
import time
from random import shuffle

# measure duration at the starting point
start_time = time.time()



################################
# Debug mode
################################
#debug_mode = True
debug_mode = False

################################
# Hyper-parameters
################################
input_length = 20 # lenght of string
train_input = ['{0:020b}'.format(i) for i in range(2 ** input_length)]
output_length = 21 # the result can be anything between 0 and 20. so it is 21
NUM_EXAMPLES = 10000 # # of train data that is 1% of the total examples
num_hidden = 24 # hidden nurons
batch_size = 1000 # an unit of gradient update
epoch = 5000 # repetition



################################
# Generate input
################################

shuffle(train_input)
train_input = [map(int, i) for i in train_input]

ti = []

for i in train_input:
    temp_list = []
    for j in i: 
        temp_list.append([j])
        
    ti.append(np.array(temp_list))
train_input = ti

if debug_mode:
    print 'train_input: '
    print train_input

################################
# Generate output
################################

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * output_length)
    temp_list[count] = 1
    train_output.append(temp_list)

if debug_mode:
    print '# of train output: ' + str(len(train_output))        
    print train_output


################################
# Generating the test data
################################

print 'Input size of total data is: ' + str(len(train_input))
print 'Output size of total data is: ' +  str(len(train_output))
print ""


test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print 'train_input: ' + str(len(train_input))
print 'test_input: ' + str(len(test_input))

print 'train_output: ' + str(len(train_output))
print 'test_output: ' + str(len(test_output))

################################
# Graph Construction
################################
import tensorflow as tf

data = tf.placeholder(tf.float32, [None, input_length, 1])
target = tf.placeholder(tf.float32, [None, output_length])


cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)

# run RNN
val, state = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)
if debug_mode:
    print 'val shape: ' + str(val.get_shape())
    print 'val shape[0]: ' + str(val.get_shape()[0])
    print 'target shape: ' + str(target.get_shape())
    print 'target shape[0]: ' + str(target.get_shape()[0])
    print 'target shape[1]: ' + str(target.get_shape()[1])


val = tf.transpose(val, [1,0,2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

if debug_mode:
    print 'val shape after transpose: ' + str(val.get_shape())
    print 'last value: ' + str(last)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target,1), tf.argmax(prediction,1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
if debug_mode:
    print 'mistakes: ' + str(mistakes)
    print 'error: ' + str(error)


init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)


no_of_batches = int(len(train_input) / batch_size)
print 'no_of_batches: ' + str(no_of_batches)


for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        if debug_mode:
            print ('inp {:2d} , out {:2d}, ptr {:2d}, batch_size {:3d} : '.format(len(inp), len(out), ptr, batch_size))
        sess.run(minimize, {data: inp, target: out})
    print "Epoch - ", str(i)
incorrect = sess.run(error, {data: test_input, target: test_output})

if debug_mode:
    print 'incorrect: ' + str(incorrect)
    print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
print('Epoch {:2d} error {:3.1f}'.format(i + 1, 100 * incorrect))
sess.close()

################################
# Measure duration
################################
end_time = time.time()
duration = end_time - start_time
print 'Start time: ' + time.ctime(start_time)
print 'End time: ' + time.ctime(end_time)
print 'Total execution time: ' + str(duration/60)












