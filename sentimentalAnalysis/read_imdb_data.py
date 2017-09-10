""" 
Author: gonsoomoon
DAte: Sep 10, 2017
Ref: Nikhil Buduma (2017). Fundamentals of deep learning. Sebastopol, CA: O’Reilly Media
"""
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='data/imdb.pkl', n_words=30000,valid_portion=0.1)

# TrainX is a list of 22,500 reviews which consists of indices of words
# The length of each review of the list varies like 25, 52, and 2500
# TrainY is a list of 22,500 sentimantal that is in the form of being negative as 0 or positive as 1
# TestX and TestY are lists of 2,500 that are the same properties as the TrainX and TrainY

trainX, trainY = train
testX, testY = test

#print ("type of trainX: ", type(trainX))
#print ("type of trainY: ", type(trainY))
# type of trainX:  <class 'list'>
print ("shape of trainX: ", np.shape(trainX))
print ("length of trainX: ", len(trainX))
#print ("trainX[0]: ", len(trainX[0]))
#print ("trainX[0]: ", len(trainX[1]))
#print ("trainX[0]: ", len(trainX[100]))
#print ("trainX[0]: ", len(trainX[1000]))

print ("shape of trainY: ", np.shape(trainY))
#print ("length of trainY: ", len(trainY))
#print ("trainX[0]: ", trainY[0])
#print ("trainX[0]: ", trainY[1])
#print ("trainX[0]: ", trainY[100])
#print ("trainX[0]: ", trainY[1000])

print ("shape of testX: ", np.shape(testX))

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=500, value=0.)
testX = pad_sequences(testX, maxlen=500, value=0.)

#print ("trainX[0]: ", trainX[0])
#print ("trainX[0]: ", trainX[1])


# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)



class IMDBDataset():
    def __init__(self, X, Y):
        self.num_examples = len(X)
        self.inputs = X
        self.tags = Y
        self.ptr = 0


    def minibatch(self, size):
        ret = None
        if self.ptr + size < len(self.inputs):
            ret =  self.inputs[self.ptr:self.ptr+size], self.tags[self.ptr:self.ptr+size]
        else:
            ret = np.concatenate((self.inputs[self.ptr:], self.inputs[:size-len(self.inputs[self.ptr:])])), np.concatenate((self.tags[self.ptr:], self.tags[:size-len(self.tags[self.ptr:])]))
        self.ptr = (self.ptr + size) % len(self.inputs)

        return ret


train = IMDBDataset(trainX, trainY)
val = IMDBDataset(testX, testY)
