# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

################################
# Debug mode
################################
#debug_mode = True
debug_mode = False


import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

################################################
# Step 1: Download the data
################################################
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Downlaod a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip',31344016)

################################################
# Read the data into a list of strings.
# Make a list of strings from zip file
################################################

def read_data(filename):
    """ Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)

# Result: 17,005,207
print('Data size', len(vocabulary))


################################################
# Step 2: Build the dictionary and replace rare words with UNK token
################################################

#vocabulary_size = 10
vocabulary_size = 50000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  # Count has the most common words by n_words -1
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  
  # Dictionary has key(word), value((length of dictionary))  
  # Ex) (Unk, 0), (the, 1), (of, 2)
  for word, _ in count:
    # Make a value in terms of length of dictionary  
    dictionary[word] = len(dictionary)      
    #if debug_mode:
    #    print ('[Debug] dictionary[word]: ' + str(dictionary[word]))
    #    print ('[Debug] word: ' + str(word))      

  data = list()
  # Out of 17,005,207, value of the word is set to index
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)

  # Result: 17,005,207
  if debug_mode:
     print ('[Debug] length of data: ' + str(len(data)))
    
  count[0][1] = unk_count
  # Swap key and value to value to key
  # Ex) (0, Unk), (1, the), (2, of)     
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

    
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

    
del vocabulary # Hint to reduce memory
print('Most common words (+UNK)', count[:5])
# Result:
# Sample data [0, 0, 0, 6, 0, 2, 0, 0, 0, 0] 
#             ['UNK', 'UNK', 'UNK', 'a', 'UNK', 'of', 'UNK', 'UNK', 'UNK', 'UNK']
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    
data_index = 0
        
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#  if debug_mode:
#     print ('[Debug] batch: ' + str(batch)) 
#     print ('[Debug] labels: ' + str(labels)) 
#     print ('[Debug] span: ' + str(span))      
     
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# Result:
# batch: [0 0 0 0 6 6 0 0]
# labels: [[0] [0] [6] [0] [0] [0] [6] [2]]
if debug_mode:
    print ('[Debug] batch: ' + str(batch)) 
    print ('[Debug] labels: ' + str(labels)) 

# Result
# 0 UNK -> 0 UNK
# 0 UNK -> 0 UNK
# 0 UNK -> 6 a
# 0 UNK -> 0 UNK
# 6 a -> 0 UNK
# 6 a -> 0 UNK
# 0 UNK -> 6 a
# 0 UNK -> 2 of

for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0],
          reverse_dictionary[labels[i,0]])

# Step 4 : build and train a skip-gram model.

batch_size = 128
embedding_size = 128    # Dimension of the embedding vector
skip_window = 1         # How many words to consider left and right
num_skips = 2           # How many times to reuse an input to generate a label

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.

valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # Context words
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # Target Word
    valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
    
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs
        embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        if debug_mode:
            print ('[Debug] embeddings: ' + str(embeddings)) 
            print ('[Debug] train_inputs: ' + str(train_inputs)) 
            print ('[Debug] embed: ' + str(embed))             
        
        # Construct the varialbes for the NCE loss
        nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels 
    # each time we evaluate the loss
    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights = nce_weights,
                           biases = nce_biases,
                           labels = train_labels,
                           inputs = embed,
                           num_sampled = num_sampled,
                           num_classes = vocabulary_size))
    
    # Construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    # Compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))

    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b = True)

    if debug_mode:
        print ('[Debug] norm: ' + str(norm)) 
        print ('[Debug] embeddings: ' + str(embeddings))         
        print ('[Debug] normalized_embeddings: ' + str(normalized_embeddings)) 
        print ('[Debug] similarity: ' + str(similarity))         
        
    # Add variable initializer
    init = tf.global_variables_initializer()
    
# Step 5: Begin training
num_steps = 100001
#num_steps = 3000

with tf.Session(graph = graph) as session:
    # We must initialize all variables before we use them
    init.run()
    print('Initialized')
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # batch_inputs        
        # [2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 2 2 1 1 0 0 0 0 3 3 1 1 0
        #  0 0 0 2 2 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 5 5 6 6 0 0 0 0 7 7 0 0 0 0
        #  0 0 0 0 0 0 0 0 0 0 7 7 0 0 1 1 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
        #  6 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]
        # batch_labels: [[0][0][2][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][1][0]
        # .....  [0] [0] [1] [0]]
        
        # We perform one update step by evaluating the optimizer
        # op (including it in the list of returned values
        # for session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss = average_loss + loss_val
        
        if step % 2000 == 0:
            if step >0:
                average_loss = average_loss /  2000
            # The average loss is an estimate of the loss over the last 2000 batches
            print('Aveerage loss at step ', step, ' :  ', average_loss)
            average_loss =0
            
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # Show similarity 
        if step % 10000 == 0:
            sim = similarity.eval()
            
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest nneighbors
                nearest = (-sim[i,:]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s: ' % valid_word
                if debug_mode:
                    print ('[Debug] valid_examples[i]: ' + str(valid_examples[i])) 
                    print ('[Debug] nearest: ' + str(nearest))                     
                
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    
# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize = (18, 18)) # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy = (x, y),
                     xytext = (5,2),
                     textcoords = 'offset points',
                     ha = 'right',
                     va = 'bottom')
    plt.savefig(filename)
    
try:
    # pylint: disabled = g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    tsne = TSNE(perplexity = 30, n_components =2, init = 'pca', n_iter = 5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
    
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    
        
        
    
        
    




