
from __future__ import print_function
import input_data
import tensorflow as tf
#from tensorflow.python import control_flow_ops
import argparse

import numpy as np

# Architecture

n_encoder_hidden_1 = 1000
n_encoder_hidden_2 = 500
n_encoder_hidden_3 = 250
n_decoder_hidden_1 = 250
n_decoder_hidden_2 = 500
n_decoder_hidden_3 = 1000


# Parameters
learning_rate = 0.01
training_epochs = 200
#training_epochs = 1
batch_size = 100
display_step = 1

# Batch Normalization
# 1. Grap the vector of logits incoming to a layer before they pass through the nonlinearity
# 2. Normalize each component of the vector of logits across all examples of the minibatch by
#    subtracting the mean and dividing by standard derivation
# 3. Given normalized inputs, (^X) use an affine transformation to restore representational power 
#    with two vectors of parameters, gamma * ^X + beta
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
    #mean, var = control_flow_ops.cond(phase_train, mean_var_with_update,lambda: (ema_mean, ema_var))
    mean, var = tf.cond(phase_train, mean_var_with_update,lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

# Apply the following to x:
# 1. x * w + bias = logits
# 2. batch_normalization(logits) = bn
# 3. sigmoid(bn)
def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))

# Take the input and compress it into a low-dimensional vector
def encoder(x, n_code, phase_train):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [784, n_encoder_hidden_1], [n_encoder_hidden_1], phase_train)
            # In the case of 1000 batch, n_encoder_hidden_1 = 10
            # x= 1000 * 784, W = 784 * 10, Bias = 10
            # hidden_1 = 1000 * 10
            # print("[Info] hidden_1 in encoder: ", hidden_1)
            #[Info] hidden_1 in encoder:  Tensor("autoencoder_model/encoder/hidden_1/Sigmoid:0", shape=(?, 10), dtype=float32)

        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_encoder_hidden_1, n_encoder_hidden_2], [n_encoder_hidden_2], phase_train)
            # hidden_1 = 1000 * 10, W = 10 * 5, Bias = 5
            # hidden_2 = 1000 * 5
            # print("[Info] hidden_2 in encoder: ", hidden_2)
            # [Info] hidden_2 in encoder:  Tensor("autoencoder_model/encoder/hidden_2/Sigmoid:0", shape=(?, 5), dtype=float32)

        with tf.variable_scope("hidden_3"):
            hidden_3 = layer(hidden_2, [n_encoder_hidden_2, n_encoder_hidden_3], [n_encoder_hidden_3], phase_train)
            # hidden_2 = 10000 * 5, W = 5 * 2, B = 2
            # hidden_3 = 1000 * 2
            # print("[Info] hidden_3 in encoder: ", hidden_3)
            # [Info] hidden_3 in encoder:  Tensor("autoencoder_model/encoder/hidden_3/Sigmoid:0", shape=(?, 2), dtype=float32)

        with tf.variable_scope("code"):
            code = layer(hidden_3, [n_encoder_hidden_3, n_code], [n_code], phase_train)
            # print("[Info] code in encoder: ", code)
            # hidden_3 = 1000 * 2, W = 2 * 3, B = 3
            # code = 1000 * 3
            #  [Info] code in encoder:  Tensor("autoencoder_model/encoder/code/Sigmoid:0", shape=(?, 2), dtype=float32)

    return code

# Invert the computation of the encoder and reconstruct the original input
def decoder(code, n_code, phase_train):
    with tf.variable_scope("decoder"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(code, [n_code, n_decoder_hidden_1], [n_decoder_hidden_1], phase_train)
            # print("[Info] hidden_1 in decoder: ", hidden_1)            
            # [Info] hidden_1 in decoder:  Tensor("autoencoder_model/decoder/hidden_1/Sigmoid:0", shape=(?, 2), dtype=float32)

        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_decoder_hidden_1, n_decoder_hidden_2], [n_decoder_hidden_2], phase_train)
            # print("[Info] hidden_2 in decoder: ", hidden_2)                        
            # [Info] hidden_2 in decoder:  Tensor("autoencoder_model/decoder/hidden_2/Sigmoid:0", shape=(?, 5), dtype=float32)            

        with tf.variable_scope("hidden_3"):
            hidden_3 = layer(hidden_2, [n_decoder_hidden_2, n_decoder_hidden_3], [n_decoder_hidden_3], phase_train)
            # print("[Info] hidden_3 in decoder: ", hidden_3)                        
            # [Info] hidden_3 in decoder:  Tensor("autoencoder_model/decoder/hidden_3/Sigmoid:0", shape=(?, 10), dtype=float32)            

        with tf.variable_scope("output"):
            output = layer(hidden_3, [n_decoder_hidden_3, 784], [784], phase_train)
            # print("[Info] output in decoder: ", output)                                    
            # [Info] output in decoder:  Tensor("autoencoder_model/decoder/output/Sigmoid:0", shape=(?, 784), dtype=float32)            

    return output

# Make a L2 norm loss function that is difference from output and input as x
def loss(output, x):
    with tf.variable_scope("training"):
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), 1))
        train_loss = tf.reduce_mean(l2)
        train_summary_op = tf.summary.scalar("train_cost", train_loss)
        return train_loss, train_summary_op

# Make a optimizer with the loss function
def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

# Make image summary on the tensorboard
def image_summary(label, tensor):
    tensor_reshaped = tf.reshape(tensor, [-1, 28, 28, 1])
    #return tf.image_summary(label, tensor_reshaped)
    return tf.summary.image(label, tensor_reshaped)

# Make evaluation function
def evaluate(output, x):
    with tf.variable_scope("validation"):
        in_im_op = image_summary("input_image", x)
        out_im_op = image_summary("output_image", output)
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x, name="val_diff")), 1))
        val_loss = tf.reduce_mean(l2)
        val_summary_op = tf.summary.scalar("val_cost", val_loss)
        return val_loss, in_im_op, out_im_op, val_summary_op


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various optimization strategies')
    parser.add_argument('n_code', nargs=1, type=str)
    args = parser.parse_args()
    n_code = args.n_code[0]

    mnist = input_data.read_data_sets("data/", one_hot=True)

    # ======================================================
    # 1. Feed input into the encoder(), producing a code
    # 2. Feed the code as embedding into the decoder()
    # 3. Make a cost function, L2 norm in the loss()
    # 4. Make an optimizer in the training()
    # 5. Make a validation loss function in evaluate()
    # 6. Create a session 
    # 7. Create a saver
    # 8. Training with loop over epoch
    #   8-1 loop over batches
    #       8-1-1   Run train_op(Gradient) per every batch
    #   8-2 Evaluate with validation set
    #   8-3 Evaluate with test set    

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            phase_train = tf.placeholder(tf.bool)

            code = encoder(x, int(n_code), phase_train)
            print ("[Info] code after encoder: " , np.shape(code))
            # For the batch size of 1000, Code shape:  (1, 1000, 2)            
            # [Info] code after encoder:  Tensor("autoencoder_model/encoder/code/Sigmoid:0", shape=(?, 2), dtype=float32)
            
            output = decoder(code, int(n_code), phase_train)
            print ("[Info] code after decoder: " , np.shape(output))            
            # [Info] code after decoder:  Tensor("autoencoder_model/encoder/code/Sigmoid:0", shape=(?, 2), dtype=float32)            

            cost, train_summary_op = loss(output, x)
            print("[Info] cost: " , cost)
            # [Info] cost:  Tensor("autoencoder_model/training/Mean:0", shape=(), dtype=float32)
            
            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x)

            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=200)

            sess = tf.Session()

            train_writer = tf.summary.FileWriter("mnist_autoencoder_hidden=" + n_code + "_logs/",
                                                graph=sess.graph)

            val_writer = tf.summary.FileWriter("mnist_autoencoder_hidden=" + n_code + "_logs/",
                                                graph=sess.graph)

            sess.run(tf.global_variables_initializer())
            
            total_batch = int(mnist.train.num_examples/batch_size)
            print("[Info] mnist.train.num_examples: ", mnist.train.num_examples)
            print("{info] total_batches: ", total_batch)
            

            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.
                # Loop over all batches
                for i in range(total_batch):
                    minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                    # print("{info] minibatch_x: ", np.shape(minibatch_x))                    

                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: minibatch_x, phase_train: True})

                    train_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                    #if i % 1000 == 0:
                        #print("[info] minibatch_x: ", minibatch_x, "minibatch_y: ", minibatch_y)                                            
                        #print("[info] minibatch_x: ", tf.size(minibatch_x), "minibatch_y: ", tf.size(minibatch_y))                                            
                        #print("[info] minibatch_x: ", tf.shape(minibatch_x), "minibatch_y: ", tf.shape(minibatch_y))                    
                        #    {info] minibatch_x:  Tensor("autoencoder_model/Shape_116:0", shape=(2,), dtype=int32) minibatch_y:  Tensor("autoencoder_model/Shape_117:0", shape=(2,), dtype=int32)
                        # Fit training using batch data
                        #print("[info] new_cost: ", new_cost)                                                                
                        #print("[info] rCode shape: ", tf.shape(rCode))                                                                                                                
                        #print("[info] train_summary: ", train_summary)                                                                                        
                    
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                    train_writer.add_summary(train_summary, sess.run(global_step))

                    validation_loss, in_im, out_im, val_summary = sess.run([eval_op, in_im_op, out_im_op, val_summary_op], 
                                                                           feed_dict={x: mnist.validation.images, phase_train: False})
                    val_writer.add_summary(in_im, sess.run(global_step))
                    val_writer.add_summary(out_im, sess.run(global_step))
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    print ("Validation Loss:", validation_loss)

                    saver.save(sess, "mnist_autoencoder_hidden=" + n_code + "_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)


            print ("Optimization Finished!")


            test_loss = sess.run(eval_op, feed_dict={x: mnist.test.images, phase_train: False})

            print ("Test Loss:", test_loss)
