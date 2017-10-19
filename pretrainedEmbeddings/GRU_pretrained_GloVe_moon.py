# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:18:27 2017

@author: tomhope
@ modified by Gonsoo Moon
"""
import zipfile
import numpy as np
import tensorflow as tf

#######################################################
# Input embeddings data
#######################################################
#path_to_glove = './data/glove.840B.300d.zip'
path_to_glove = './data/glove.6B.zip'

#######################################################
# Hyperparameter
#######################################################
PRE_TRAINED = True
#PRE_TRAINED = False
GLOVE_SIZE = 50 # embedding dimension
batch_size = 128
embedding_dimension = 50
num_classes = 2
hidden_layer_size = 32
times_step = 6

input_num = 10000 # number of sentences to be generated
training_steps = 20000 # number of training steps

#######################################################
# Generating Input
#######################################################
digit_to_word_map = {1:"one",2:"two", 3:"three", 4:"four", 5:"five",6:"six",7:"seven",8:"eight",9:"nine"}
digit_to_word_map[0] = "PAD_TOKEN"

even_sentences = []
odd_sentences = []
seqlens = []

# Create two kinds of sentences - sequences of odd and even digits.
for i in range(input_num):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    rand_odd_ints = np.random.choice(range(1,10,2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2,10,2), rand_seq_len)

    if rand_seq_len < times_step:
        rand_odd_ints = np.append(rand_odd_ints, [0] * (times_step - rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0] * (times_step - rand_seq_len))
        
    even_sentences.append(" ".join(digit_to_word_map[r] for
                                   r in rand_odd_ints))
    odd_sentences.append(" ".join(digit_to_word_map[r] for 
                                  r in rand_even_ints))
    
#print ("even_sentences: ", even_sentences)
#print ("odd_sentences: ", odd_sentences)
#even_sentences:  ['Three One Five One Nine Seven', 'Seven Three Nine PAD_TOKEN PAD_TOKEN PAD_TOKEN', 'One Five Seven Seven Five Seven', 'One One Five Seven PAD_TOKEN PAD_TOKEN']
#odd_sentences:  ['Six Eight Eight Six Four Two', 'Six Eight Four PAD_TOKEN PAD_TOKEN PAD_TOKEN', 'Two Six Eight Eight Six Six', 'Two Six Two Four PAD_TOKEN PAD_TOKEN']


data = even_sentences + odd_sentences

seqlens *= 2 # odd + even
# Generate labels
labels = [1] * input_num + [0] * input_num
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0] * 2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding
    
#print ('labels', labels)
#labels [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]

# With the data, make a word2index map
word2index_map = {}
index = 0
for sent in data:
    for word in sent.split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
            
# Make index2word_map with the word2index map
index2word_map = {index : word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)
print ("vocabulary_size: ", vocabulary_size)
#print ("index2word_map: ", index2word_map)
#print ("word2index_map: ", word2index_map)
#print ("vocabulary_size: ", vocabulary_size)
# index2word_map:  {0: 'One', 1: 'Seven', 2: 'Three', 3: 'Nine', 4: 'Five', 5: 'PAD_TOKEN', 6: 'Four', 7: 'Two', 8: 'Eight', 9: 'Six'}
# word2index_map:  {'One': 0, 'Seven': 1, 'Three': 2, 'Nine': 3, 'Five': 4, 'PAD_TOKEN': 5, 'Four': 6, 'Two': 7, 'Eight': 8, 'Six': 9}
#vocabulary_size:  10

#######################################################
# Extract embeddings only for digits in the word2index_map 
# from the glove zip file
# Finally, make embedding matrix with an index of word composing of voc * embedding dimension
#######################################################
def get_glove(path_to_glove, word2index_map):
    embedding_weights = {}
    count_all_words = 0
    with zipfile.ZipFile(path_to_glove) as z:
        with z.open('glove.6B.50d.txt') as f:
            for line in f:
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                #print('word: ', word)
                if word in word2index_map:
                    #print(word)
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype = 'float32')
                    #print('coefs: ', coefs)
                    coefs = coefs / np.linalg.norm(coefs)
                    #print('normalized coefs: ', coefs)                    
                    embedding_weights[word] = coefs
                if count_all_words == len(word2index_map) -1:
                    break
    
    return embedding_weights

# Generate word2embedding dictionary
word2embedding_dict = get_glove(path_to_glove, word2index_map)
#print('word2embedding_dict: ', word2embedding_dict)
# The word 'nine' has 50 embedding dimension
#'nine': array([ 
#        0.02368277,  0.019758  ,  0.11896203,  0.02049459,  0.08939567,
#        0.11061854, -0.18064567,  0.04335211, -0.05064871, -0.0788634 ,
#       -0.10507336, -0.17544484, -0.12045223,  0.0694176 ,  0.12010985,
#       -0.19633326, -0.0211912 , -0.08969054, -0.24844444, -0.01839961,
#       -0.00145708,  0.06725653,  0.15522534, -0.02781889, -0.08887915,
#       -0.14246272,  0.06303135, -0.13568859, -0.0805416 , -0.06879026,
#        0.68485492,  0.09556819,  0.0665619 ,  0.08483406,  0.2033627 ,
#        0.07244945,  0.08415724,  0.02455749, -0.02101903,  0.01568362,
#       -0.19343996, -0.02069843,  0.12602708,  0.04019956, -0.07120267,
#        0.02607341, -0.09686048, -0.01766362, -0.04628698, -0.16547659], 
#        dtype=float32)}

# initialize embedding_matrix with 0
embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE))
# embedding matrix shape:  (10, 50)

# Make an embedding matrix with word2embedding_dict and word2index_map
for word, index in word2index_map.items():
    if not word == "PAD_TOKEN":
        word_embedding = word2embedding_dict[word]
        embedding_matrix[index,:] = word_embedding

#print("embedding_matrix: ", embedding_matrix)
print("embedding_matrix shape: ", embedding_matrix.shape)
# embedding_matrix shape:  (10, 50)

#######################################################
# Function: 
#    Compute cosine distance
# Arg:
# target: what is compared to other numbers
# normalized_embeddings_matrix:  
# normalized_embeddings_matrix,word2index_map
#######################################################

def computeCosineDistance(target, normalized_embeddings_matrix,word2index_map, index2word_map):              
    ref_word = normalized_embeddings_matrix[word2index_map[target]]
    #print("ref_word: ", ref_word)
    #print("ref_word: ", np.shape(ref_word))
    # ref_word:  (50,)
    
    cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
    #cosine_dists:  [ 0.98951083  0.86495638  0.9708268          nan  0.9813419   1.00000012
    #  0.98581821  0.97698307  0.99423319  0.98859024]
    # cosine_dists shape:  (10,)
    #print("cosine_dists: ", cosine_dists)
    #print("cosine_dists: ", np.shape(cosine_dists))
    
    ff = np.argsort(cosine_dists)[::-1][0:10]
    #print("ff: ", ff)
    for f in ff:
        print(index2word_map[f],"\t", cosine_dists[f])




#######################################################
# Prepare for a train and test data set
#######################################################
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
#print('data_indices: ', data_indices)

# shuffle data
data = np.array(data)[data_indices]
#print('data: ', data)
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]

train_x = data[:input_num]
train_y = labels[:input_num]
train_seqlens = seqlens[:input_num]

# print("train_x: ", train_x)
# train_x:  [
#  'three three three PAD_TOKEN PAD_TOKEN PAD_TOKEN'
#  'six eight two two PAD_TOKEN PAD_TOKEN'
#  'one five five one PAD_TOKEN PAD_TOKEN'
# print("train_y: ", train_y)
# train_y:  
#  [[0 1]
#   [1 0]
#   [0 1]
# print("train_seqlens: ", train_seqlens)
# train_seqlens:  [3 4 4]

test_x = data[input_num:]
test_y = labels[input_num:]
test_seqlens = seqlens[input_num:]


#######################################################
# Prepare for a train and test data set
# After shuffling indices of the data, extract a batch from 0 to number of the batch_size
#######################################################
def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    # Shuffle indices of the data_x
    np.random.shuffle(instance_indices)
    # Extract data by the batch_size with the shuffled indices
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens

x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                    train_x, train_y,
                                                    train_seqlens)
# For the x_batch, '3' means PAD_TOKEN
# x_batch:  [[6, 7, 7, 8, 7, 3], [4, 1, 1, 3, 3, 3], [8, 6, 6, 3, 3, 3], [1, 1, 2, 4, 3, 3]]
# y_batch:  [array([1, 0]), array([0, 1]), array([1, 0]), array([0, 1])]
# seqlen_batch:  [5, 3, 3, 4]

#######################################################
# Placeholder for an input to be fed into the GRU network
#######################################################
# x' value
_inputs = tf.placeholder(tf.int32, shape = [batch_size, times_step])
# This is for the pretrained embedding values
embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, GLOVE_SIZE])
# y' value
_labels = tf.placeholder(tf.float32, shape = [batch_size, num_classes])
# sequence length except for "PAD_TOKEN"
_seqlens = tf.placeholder(tf.int32, shape = [batch_size])

#######################################################
# Embedding layer
# Drawing graph: 
#   inputs(32,6), embeddings(10, 50) --> embed(32, 6, 50)
#   embedding_placeholder --> embeddings --> embedding_init
#######################################################
if PRE_TRAINED:
    print("Use of Pretrained Embeddings")
    embeddings = tf.Variable(tf.constant(0.0, shape = [vocabulary_size, GLOVE_SIZE ]), trainable = True)
    # embeddings = tf.Variable(tf.constant(0.0, shape = [vocabulary_size, GLOVE_SIZE ]), trainable = False)    
    embedding_init = embeddings.assign(embedding_placeholder)
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
    
else:
    print("Not using Pretrained Embeddings")    
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_dimension],
                              -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
    # Print cosine distance with a target word
    computeCosineDistance("three", embeddings,word2index_map, index2word_map)        
    # embed shape [batch_size, sequence_steps, embedding dimension]
    # For a batch size of 32, a sequence length of 6 and an embedding dimension of 50,
    # its shape is (32, 6, 50)
    
#######################################################
# Bidirectional GRU and Dropout layer
# Drawing graph: 
#   inputs, embeddings --> embed, seqlens, gru_fw_cell, gru_bw_cell --> 
#   --> outputs (2, 32, 6, 48), states (2, 32, 48)
# Output shape: [layers, batch_size, sequence_length, hidden_layer_size]
# states shape: [layers, batch_size, hidden_layer_size ]
# For the layers of the states shape, they are forward and backward state vectors
#######################################################    
with tf.name_scope("biGRU"):
    with tf.variable_scope('forward'):
        gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)
        
    with tf.variable_scope('backward'):
        gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)
        
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell,
                                                          cell_bw = gru_bw_cell,
                                                          inputs = embed,
                                                          sequence_length = _seqlens,
                                                          dtype = tf.float32,
                                                          scope = "biGRU")

#######################################################
# Linear Layer
# Drawing graph: 
#   inputs, embeddings --> embed, seqlens, gru_fw_cell, gru_bw_cell --> 
#   --> outputs (2, 32, 6, 48), states (2, 32, 48)
#   --> states2 [32, 96)] --> final_output (32, 2)
# states2 shape: [batch_size, hidden_layer_size ]
# final_output shape: [batch_size, num_classes]
#######################################################    

# Concatenate forward and backward state vectors
states2 = tf.concat(values = states, axis= 1 )

# Multiply hidden_layer_size by 2 because of incorporating forward and backward passes
weights = {
        'linear_layer' : tf.Variable(tf.truncated_normal([ 2 * hidden_layer_size,
                                                          num_classes],mean=0, stddev = .01))}
print("weights shape: ", np.shape(weights))
    
biases = {
        'linear_layer': tf.Variable(tf.truncated_normal([num_classes],mean=0, stddev = .01))}


#extract the final state and use in a linear layer
final_output = tf.matmul(states2, weights['linear_layer']) + biases['linear_layer']


#######################################################
# Softmax and Cross Entropy Layer
# Drawing graph: 
#   inputs, embeddings --> embed, seqlens, gru_fw_cell, gru_bw_cell --> 
#   --> outputs (2, 32, 6, 48), states (2, 32, 48)
#   --> states2 [32, 96)] --> final_output (32, 2)
#       --> cross_entropy --> train_step
#       --> correct_prediction with _labels --> accuracy
#######################################################    
softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output,
                                                   labels = _labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

#######################################################
# Run graph
#######################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    if PRE_TRAINED:
        print("Run embedding_init")
        sess.run(embedding_init, feed_dict = {embedding_placeholder: embedding_matrix})
        # Print cosine distance with a target word
        computeCosineDistance("three", embedding_matrix,word2index_map, index2word_map)    
    else:
        sess.run(embeddings, feed_dict = {embedding_placeholder: embedding_matrix})        
        # Print cosine distance with a target word
        computeCosineDistance("three", embedding_matrix,word2index_map, index2word_map)    
        
        
    
    for step in range(training_steps):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                            train_x, train_y,
                                                            train_seqlens)
        
        #print('x_batch: ', x_batch)
        #print('y_batch: ', y_batch)
        #print('seqlen_batch: ', seqlen_batch)
        
        
        sess.run(train_step, feed_dict = {_inputs:x_batch, _labels: y_batch, _seqlens:seqlen_batch})
        r_outputs, r_states, r_states2, r_final_output, r_embeddings = sess.run(
                [outputs, states,states2, final_output, embeddings], feed_dict = {_inputs:x_batch, _labels: y_batch, _seqlens:seqlen_batch})        

        #print("r_outputs: ", np.shape(r_outputs))
        #print("r_states: ", np.shape(r_states))
        #print("r_states2: ", np.shape(r_states2))        
        #print("final_output: ", np.shape(r_final_output))                
        #print("embedding shape:", r_embeddings.shape)        

        #print("r_states: ", r_states)
        #
        # print("r_states2: ", r_states2)        
        
        if step % 100 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict = {_inputs: x_batch,
                                                  _labels:y_batch,
                                                  _seqlens:seqlen_batch})
            print("Loss, Accuracy at %d, %d: %.5f" % (step,loss, acc))
            
    
    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens)
        
        #print('x_batch_test: ', x_test)
        #print('y_batch_test: ', y_test)
        #print('seqlen_batch_test: ', seqlen_test)
        
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1),accuracy],
                                        feed_dict = {_inputs: x_test, 
                                                     _labels: y_test,
                                                     _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
        
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)
    print("normalized_embeddings_matrix shape: ", np.shape(normalized_embeddings_matrix))
    # normalized_embeddings_matrix shape:  (10, 50)
    #print("normalized_embeddings_matrix : ", normalized_embeddings_matrix)
    # normalized_embeddings_matrix shape:  (10, 50)
    
# Print cosine distance with a target word
computeCosineDistance("three", normalized_embeddings_matrix,word2index_map, index2word_map)    
    















        
        
        








    
    




