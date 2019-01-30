import pickle
import numpy
numpy.random.seed(123)
import os
os.chdir("/Users/gonsoomoon/Documents/DeepLearning/kaggle/rossmann/rossmann_publish")

from models import Model_Util
from models import *

import sys
sys.setrecursionlimit(10000)

class TRAIN_DRIVER():
    def __init__(self):
        FEATURE_TRAIN_DATA = './data/feature_train_data.pickle'
        train_ratio = 0.8
        # shuffle_data = False
        # one_hot_as_input = False
        # embeddings_as_input = False
        # save_embeddings = True
        # saved_embeddings_fname = "embeddings.pickle"  # set save_embeddings to True to create this file

        f = open(FEATURE_TRAIN_DATA, 'rb')
        (X, y) = pickle.load(f)

        num_records = len(X)
        train_size = int(train_ratio * num_records)

        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        # X_train, y_train = sample(X_train, y_train, 200000)  # Simulate data sparsity
        print("Number of samples used for training: " + str(y_train.shape[0]))

        models = []

        print("Fitting NN_with_EntityEmbedding...")
        model_util = Model_Util()
        for i in range(1):
            models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val, model_util))

        print("Evaluate combined models...")
        print("Training error...")
        r_train = self.evaluate_models(models, X_train, y_train)
        print(r_train)

        # Saved model to disk
        # Result on validation data:  0.13954197912027022

        print("Validation error...")
        r_val = self.evaluate_models(models, X_val, y_val)
        print(r_val)

    def sample(self, X, y, n):
        '''random samples'''
        num_row = X.shape[0]
        indices = numpy.random.randint(num_row, size=n)
        #return X[indices, :], y[indices]
        return X[0:n, :], y[0:n]

    def evaluate_models(self, models, X, y):
        assert(min(y) > 0)
        guessed_sales = numpy.array([model.guess(X) for model in models])
        print('guessed_sales: {}'.format(guessed_sales))
        mean_sales = guessed_sales.mean(axis=0)
        relative_err = numpy.absolute((y - mean_sales) / y)
        result = numpy.sum(relative_err) / len(y)
        return result


# TRAIN_DRIVER()