""" 
Author: gonsoomoon
DAte: Sep 10, 2017
Ref: Nikhil Buduma (2017). Fundamentals of deep learning. Sebastopol, CA: O’Reilly Media
"""


import tensorflow as tf
#from lstm import LSTMCell
import read_imdb_data as data
import numpy as np


#training_epochs = 1000
training_epochs = 1
batch_size = 32
display_step = 1

# Create embeddings
def embedding_layer(input, weight_shape):
    # input: 32 * 500, weight_shape: 30,000 * 512
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    E = tf.get_variable("E", weight_shape,
                        initializer=weight_init)
    incoming = tf.cast(input, tf.int32)
    embeddings = tf.nn.embedding_lookup(E, incoming)
    # embedding shape:  (32, 500, 512)    
    return embeddings


# Create dropout LSTM and then provide output and state
def lstm(input, hidden_dim, keep_prob, phase_train):
        #lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        lstm = tf.contrib.rnn.BasicLSTMCell(hidden_dim)        

        #dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * 2, state_is_tuple=True)
        lstm_outputs, state = tf.nn.dynamic_rnn(dropout_lstm, input, dtype=tf.float32)
        #return tf.squeeze(tf.slice(lstm_outputs, [0, tf.shape(lstm_outputs)[1]-1, 0], [tf.shape(lstm_outputs)[0], 1, tf.shape(lstm_outputs)[2]]))
        # Extract the last output from the sequence of the outputs
        return tf.reduce_max(lstm_outputs, reduction_indices=[1])

# Batch normalization
def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

# Create a neural network layer with activation function, which is sigmoid, and batch normalization 
def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))


# Data flow with this network
# input 32 * 500 -> embedding  --> 32 * 500 * 512 --> lstm
# --> 32 * 512 --> sigmoid --> 32 * 2 as output

def inference(input, phase_train):
    # input: 32 * 500
    
    # Create embeddings
    # embedding shape:  (32, 500, 512)
    embedding = embedding_layer(input, [30000, 512])    

    # With the embeddings as an input for lstm, an output of the lstm is provided
    # Sequence lenght: 512, dropout rate: 0.5
    # lstm output shape:  (32, 512)
    lstm_output = lstm(embedding, 512, 0.5, phase_train)
    
    
    # Pass in the lstm output to the neural network layer
    # outshape: 32 * 2
    output = layer(lstm_output, [512, 2], [2], phase_train)
    return output, embedding, lstm_output

# Create loss function with the output of the lstm network and provide a loss
def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y)
    #tf.nn.softmax_cross_entropy_with_logits(logits = yPredbyNN, labels=Y)
    loss = tf.reduce_mean(xentropy)
    #train_loss_summary_op = tf.scalar_summary("train_cost", loss)
    train_loss_summary_op = tf.summary.scalar("train_cost", loss)    
    val_loss_summary_op = tf.summary.scalar("val_cost", loss)
    return loss, train_loss_summary_op, val_loss_summary_op

# With the cost function, define an optimizer
def training(cost, global_step):
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')    
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

# Define an evaluation function
def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy)
    return accuracy, accuracy_summary_op

if __name__ == '__main__':

    with tf.Graph().as_default():
        #with tf.device('/gpu:0'):
        with tf.device('/cpu:0'):  
            # Define placeholder 
            x = tf.placeholder("float", [None, 500])
            y = tf.placeholder("float", [None, 2])
            phase_train = tf.placeholder(tf.bool)


            # Generate output of a LSTM network
            output, embedding, lstm_output = inference(x, phase_train)

            # Generate cost
            cost, train_loss_summary_op, val_loss_summary_op = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Genearte an optimizer
            train_op = training(cost, global_step)

            # Define an evaluation op.
            eval_op, eval_summary_op = evaluate(output, y)

            saver = tf.train.Saver(max_to_keep=100)

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

            summary_writer = tf.summary.FileWriter("imdb_lstm_logs/",
                                                graph=sess.graph)

            init_op = tf.initialize_all_variables()

            sess.run(init_op)

            for epoch in range(training_epochs):

                avg_cost = 0.
                print ("data.train.num_examples: ", data.train.num_examples)
                total_batch = int(data.train.num_examples/batch_size)
                print ("Total of %d minbatches in epoch %d" % (total_batch, epoch))
                # Loop over all batches
                #for i in range(total_batch):                
                for i in range(1):
                    minibatch_x, minibatch_y = data.train.minibatch(batch_size)
                    print ("minibatch_x: ", np.shape(minibatch_x), " minibatch_y: ", np.shape(minibatch_y))
                    #minibatch_x:  (32, 500)  minibatch_y:  (32, 2)
                    #minibatch_x:  [[ 17  25  10 ...,   0   0   0]
                    # [ 16   1  32 ...,   0   0   0]
                    # [  1   2   1 ...,   0   0   0]
                    # ..., 
                    # [ 57  30   1 ...,   0   0   0]
                    # [  1 106 236 ...,   0   0   0]
                    # [  1   1  15 ...,   0   0   0]]
                    # minibatch_y:  [[ 1.  0.]
                    # [ 1.  0.]
                    # [ 1.  0.]
                    # [ 1.  0.]
                    # [ 0.  1.]
                    # [ 1.  0.]
                    # [ 0.  1.]
                    
                    
                    # Fit training using batch data
                    #_, new_cost, train_summary = sess.run([train_op, cost, train_loss_summary_op], feed_dict={x: minibatch_x, y: minibatch_y, phase_train: True})
                    _, new_cost, train_summary, r_output, r_embedding, r_lstm_output = sess.run([train_op, cost, train_loss_summary_op, output, embedding, lstm_output], 
                                                                                                feed_dict={x: minibatch_x, y: minibatch_y, phase_train: True})
                    
                    print("embedding shape: ", np.shape(r_embedding))
                    print("lstm output shape: ", np.shape(r_lstm_output))
                    print("output shape: ", np.shape(r_output))
                    #output shape:  (32, 2)
                    summary_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                    print ("Training cost for batch %d in epoch %d was:" % (i, epoch), new_cost)
                    if i % 100 == 0:
                        print ("Epoch:", '%04d' % (epoch+1), "Minibatch:", '%04d' % (i+1), "cost =", "{:.9f}".format((avg_cost * total_batch)/(i+1)))
                        val_x, val_y = data.val.minibatch(data.val.num_examples)
                        val_accuracy, val_summary, val_loss_summary = sess.run([eval_op, eval_summary_op, val_loss_summary_op], feed_dict={x: val_x, y: val_y, phase_train: False})
                        summary_writer.add_summary(val_summary, sess.run(global_step))
                        summary_writer.add_summary(val_loss_summary, sess.run(global_step))
                        print ("Validation Accuracy:", val_accuracy)

                        saver.save(sess, "imdb_lstm_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)


            print ("Optimization Finished!")
