"""
Created on Sat Jul 29 16:27:42 2017

@author: gonsoomoon
"""

import tensorflow as tf
import numpy as np


tf.set_random_seed(777) # reproducibility


class StockPrediction(object):
    # Hyperparameters are set
    def __init__(self, inputFile,outputFile, input_dim, trainTestRatio, sequence_length, epoch):
    
        self.inputFile = inputFile  # Input data
        self.outputFile = outputFile # Save target, prediction and difference
        self.input_dim = input_dim # # 5 features: Open High Low Volume Close
        self.output_dim = 1 # Predict only "Close"
        self.learning_rate  = 0.01        
        self.sequence_length = sequence_length # number of sequences
        self.epoch = epoch
        self.hidden_size = 10 # # of neurons
        self.trainTestRatio = trainTestRatio # train and test ratio
        self.printThreshold = 20 # A frequence of printing out "loss" value

    def run(self):
        dataX, dataY = self.readInput() # read input file and make batches
        self.splitTrainTestData(dataX, dataY) # split train and test 
        self.printAttributes() # print out hyperparameters
        rmse, originalTestY,originalTestPredict, difference = self.train() # train and predict
        
        print("RMSE: {}".format(rmse))
        print("\n")
        # save target, prediction and difference
        self.writeCSV(originalTestY, originalTestPredict, difference, self.outputFile)                
        

        
        
    def printAttributes(self):
        print('[Info] inputFile: ', self.inputFile)    
        print('[Info] outputFile: ', self.outputFile)            
        print('[Info] input_dim: ', self.input_dim)                    
        print('[Info] learning_rate: ', self.learning_rate)                            
        print('[Info] hidden_size: ', self.hidden_size)                    
        print('[Info] sequence_length: ', self.sequence_length)                            
        print('[Info] epoch: ', self.epoch)                                    
        print('[Info] trainTestRatio: ', self.trainTestRatio)                                            
        print ('[Info] batch size: %d' % (self.batch_size))         
        
    # Read data from the input file    
    def readInput(self):
        # Open, High, Low, Volume, Close
        xy = np.loadtxt(self.inputFile, delimiter=',')
        print('[Info] data shape', xy.shape)
        # [Info] data shape (a, b). e.g. (732, 5)

        #xy = xy[::-1] # reverse order (chronically ordered)
        # labelMin and labelMax are used to denormalize
        scaleXY, labelMin, labelMax = self.MinMaxScaler(xy)
        x = scaleXY
        y = scaleXY[:, [-1]] # Close as label
        # e.g.
        # ('x shape:', (732, 5))
        # ('y shape:', (732, 1))
        
        self.labelMin = labelMin
        self.labelMax = labelMax
        
        dataX, dataY = self.batchData(x,y)
        return dataX, dataY

    # Make batches
    def batchData(self, x, y):
        # build a dataset with 1 of moving window
        dataX = []
        dataY = []
        # for x:
        # 36	37	38	39	40
        # 31	32	33	34	35
        # 26	27	28	29	30        
        # 21	22	23	24	25        
        # 16	17	18	19	20        
        # 11	12	13	14	15
        # 6	7	8	9	10
        # 1	2	3	4	5                
        #
        # For y
        # [[ 40.]
        # [ 35.]
        # [ 30.]
        # [ 25.]
        # [ 20.]
        # [ 15.]
        # [ 10.]
        #[  5.]]
        #
        # len(y): 8, sequence_length: 3 : 5 batches
        #(array([[ 36.,  37.,  38.,  39.,  40.],
        #       [ 31.,  32.,  33.,  34.,  35.],
        #       [ 26.,  27.,  28.,  29.,  30.]]), '->', array([ 25.]))
        #(array([[ 31.,  32.,  33.,  34.,  35.],
        #       [ 26.,  27.,  28.,  29.,  30.],
        #       [ 21.,  22.,  23.,  24.,  25.]]), '->', array([ 20.]))
        #(array([[ 26.,  27.,  28.,  29.,  30.],
        #       [ 21.,  22.,  23.,  24.,  25.],
        #       [ 16.,  17.,  18.,  19.,  20.]]), '->', array([ 15.]))
        #(array([[ 21.,  22.,  23.,  24.,  25.],
        #       [ 16.,  17.,  18.,  19.,  20.],
        #       [ 11.,  12.,  13.,  14.,  15.]]), '->', array([ 10.]))
        #(array([[ 16.,  17.,  18.,  19.,  20.],
        #       [ 11.,  12.,  13.,  14.,  15.],
        #       [  6.,   7.,   8.,   9.,  10.]]), '->', array([ 5.]))
    
        # Make a batch by the sequence length
        for i in range(0, len(y) - self.sequence_length):
            _x = x[i: i + self.sequence_length ]
            _y = y[i + self.sequence_length] # Next close price
            #print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)
            
        self.batch_size = len(dataX)
        
        return dataX, dataY
    
    # Split dataX, dataY into trainX, testX, trainY and testY depending on the trainTestRatio
    def splitTrainTestData(self,dataX, dataY):

        # train/test split
        train_size = int(len(dataY) * self.trainTestRatio)
        #test_size = len(dataY) - train_size
        # The testX batches are chosed from the last point of the trainX
        # For example, batches = {1 2 3 4 5 6 7 8 9 10} with 0.9 of the trainTestRatio,
        # 1 to 9 batches are assigned to trainX, 10 is assigned to the testX
        trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
        print("[Info] trainX shape: {} ".format(trainX.shape))
        print("[Info] testX shape: {} ".format(testX.shape))

        # e.g.
        # ('[Info] trainX: ', (507, 7, 5))
        # -> 507 batches, 7 sequence length, 5 input dimension (5 features)
        # ('[Info] testX: ', (218, 7, 5))
        # -> 218 batches, 7 sequence length, 5 input dimension (5 features)
        
        
        trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])
        print("[Info] trainY shape: {} ".format(trainY.shape))
        print("[Info] testY shape: {} ".format(testY.shape))

        # e.g.
        # ('[Info] trainX: ', (507, 1))
        # -> 507 batches, 1 output dimension
        # ('[Info] testX: ', (218, 1))
        # -> 218 batches, 1 output dimension
        
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        
        
    def train(self):

    #==============================================
    # Build LSTM and FC graph
    # 1. Define X, Y placeholder
    # 2. Define basic LSTM with hidden_size that is # of neurons
    # 3. Run LSTM  with X placeholder
    # 4. Reshape outputs from the LSTM to feed in the FC layer
    #==============================================
        
        
        # input placeholders
        X = tf.placeholder(tf.float32, [None, self.sequence_length, self.input_dim])
        Y = tf.placeholder(tf.float32, [None, 1])
        
        # build a LSTM network
        cell = tf.contrib.rnn.BasicLSTMCell(
                num_units = self.hidden_size, # 10 
                state_is_tuple = True,
                activation = tf.tanh)
        
        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
        #print('[Info] outputs shape after RNN:', outputs.shape)
        # For the X = [2,4,5]: 2 of batches, 4 of sequence length and 5 of input dimensions
        #('outputs shape\n', (2, 4, 10)): 2 of batches, 4 of sequence length and 10 of outputs
        
        Y_pred = tf.contrib.layers.fully_connected(
                outputs[:, -1], # [2,10]
                self.output_dim, # 1
                activation_fn = None) # We use the last cell's output

        # For the outputs[:,-1]        
        # From the outputs, (2,4,10), extract the last output row per batch.
        # In other words, from the sequence, the last element is chosen.
        # For example, t= [2, 4, 10] of
        # array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        #        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
        #
        #       [[40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        #        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        #        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        #        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]]])
            
        # t[:,-1]
        # array([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]])    
        
        print(['Info Y_pred shape', Y_pred.shape])
        # Y_pred shape: [None, 1]
        
        
        #==============================================
        # Build loss function graph
        # 1. Define loss function
        # 2. Define optimizer
        # 3. For test, define targets and predictions placeholder
        # 4. Define RMSE error function for measurement
        # 5. Run the step, training
        # 6. Run the step, test
        # 7. On the test, Revert the results to orignial values
        # 8. On the test, compute RMSE with the original values
        #==============================================
        
        # cost / loss
        loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)
        
        # RMSE
        targets = tf.placeholder(tf.float32, [None,1])
        predictions = tf.placeholder(tf.float32, [None,1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
                    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # =============================================
            # Training step
            # =============================================            
            for i in range(self.epoch):
                
                _, step_loss, result_outputs = sess.run([train, loss, outputs], 
                                        feed_dict = {X: self.trainX, Y: self.trainY})
                if i % self.printThreshold == 0 :
                    print("[step: {}] loss: {}".format(i, step_loss))
                    
            # =============================================
            # Test step
            # =============================================            
                    
            # Predict Y as test_predict with the testX
            test_predict = sess.run(Y_pred, feed_dict={X: self.testX})
            # revert target value as testY to original value
            originalTestY = self.revertMinMaxScaler(self.testY, self.labelMin, self.labelMax)
            # revert the prediction, Y, to original value
            originalTestPredict = self.revertMinMaxScaler(test_predict, self.labelMin, self.labelMax)    
            # calculate a difference
            difference = originalTestY - originalTestPredict

            # Compute RMSE with the orignial values
            rmse_val= sess.run(rmse, feed_dict= {targets: originalTestY, predictions: originalTestPredict})                

        return rmse_val, originalTestY, originalTestPredict, difference
            
    # Save the target and prediction to csv file.
    def writeCSV(self, originalY,originalPredict, difference, csvFile):
        
        #========================================================
        # Save to csv file
        #========================================================    
        import csv        

        # reshape rank back by 1           
        originalTestYBack = np.reshape(originalY,[-1])
        originalTestPredictBack = np.reshape(originalPredict,[-1])             
        differenceBack = np.reshape(difference,[-1])                 
        

        with open(csvFile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in zip(originalTestYBack, originalTestPredictBack, differenceBack):    
                writer.writerow(val)
             

    # MinMax normalization    
    def MinMaxScaler(self,data):
        """    
        References
        ----------
        .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        """
        
        numerator = data - np.min(data,0)
        denominator = np.max(data,0) - np.min(data,0)

        # For denormalization, minimum and maximus values are returned on the label as Y
        labelMin = np.min(data[:,-1],0)
        labelMax = np.max(data[:,-1],0)    

        # noise term prevents the zero division  for the 1e-7
        return numerator / (denominator + 1e-7), labelMin, labelMax    


    # denormalization
    def revertMinMaxScaler(self, scaledValue, minValue, maxValue):
        originalValue = scaledValue * (maxValue - minValue + 1e-7) + minValue
        
        return originalValue
            
def main(inputFile, outputFile, trainTestRatio, sequence_length, epoch):
    # Create rnn object
    rnn = StockPrediction(inputFile, outputFile, 5, trainTestRatio, sequence_length, epoch)
    rnn.run()
    
import sys
    
if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    trainTestRatio = sys.argv[3]
    sequence_length = sys.argv[4]
    epoch = sys.argv[5]
    main(
            inputFile,
            outputFile,
            float(trainTestRatio),
            int(sequence_length),
            int(epoch))


        
    
    
    
















 




    
    
    
    
    

