import pickle
import numpy as np
np.random.seed(123)
import sys
sys.setrecursionlimit(10000)
from models import NN_with_EntityEmbedding_Loading, Model_Util

class Inference():
    def __init__(self):

        self.FEATURE_TRAIN_DATA = './data/feature_train_data.pickle'
        self.SAVED_MODEL_JSON = "./weight/saved_model_json.json"
        self.SAVED_MODEL_WEIGHT = "./weight/saved_model_weight.h5"

    def inference_val_data(self):

        train_ratio = 0.9
        f = open(self.FEATURE_TRAIN_DATA, 'rb')
        (X, y) = pickle.load(f)

        num_records = len(X)
        train_size = int(train_ratio * num_records)


        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        X_train, y_train = self.sample(X_train, y_train, 200000)  # Simulate data sparsity
        print("Number of samples used for training: " + str(y_train.shape[0]))

        model_util = Model_Util()
        NN_E = NN_with_EntityEmbedding_Loading(self.SAVED_MODEL_JSON,self.SAVED_MODEL_WEIGHT , y_train, y_val, model_util)

        models = list()
        models.append(NN_E)

        print("Evaluate combined models...")
        print("Training error...")
        r_train = self.evaluate_models(models, X_train, y_train)
        print(r_train)

        print("Validation error...")
        r_val = self.evaluate_models(models, X_val, y_val)
        print(r_val)

        # When sample = 675,470
        # 0.06575388987219577
        # Validation error...
        # guessed_sales: [[ 4428.5576 10174.639   6026.1987 ...  8002.397   5885.008   5507.6406]]
        # 0.09761840799543625

    def inference_no_storeid(self):
        train_ratio = 0.9
        f = open(self.FEATURE_TRAIN_DATA, 'rb')
        (X, y) = pickle.load(f)

        num_records = len(X)
        train_size = int(train_ratio * num_records)

        X_test = X[-1]
        y_test = y[-1]

        X_test_nostore = np.array([0, 20, 4, 1, 2, 9, 24, 4])
        y_test_nostore = y[-1]

        y_train = y[:train_size]
        y_val = y[train_size:]

        model_util = Model_Util()
        NN_E = NN_with_EntityEmbedding_Loading(self.SAVED_MODEL_JSON,self.SAVED_MODEL_WEIGHT , y_train, y_val, model_util)

        models = list()
        models.append(NN_E)

        print("Test error...")
        r_test = self.evaluate_models_nostore_id(models, X_test, y_test)
        print(r_test)

        print("With no store id, Test error...")
        r_test_nostore = self.evaluate_models_nostore_id(models, X_test_nostore, y_test_nostore)
        print(r_test_nostore)

        # Test error
        # ground_true: 5263 , guessed_sales: [[5507.6406]], diff: [[-244.64062]]
        # 0.046483114
        # With differnt store id, Test error...
        # ground_true: 5263 , guessed_sales: [[7423.2456]], diff: [[-2160.2456]]
        0.41045898

    def sample(self, X, y, n):
        '''random samples'''
        return X[0:n, :], y[0:n]

    def evaluate_models(self, models, X, y):
        assert(min(y) > 0)
        guessed_sales = np.array([model.guess(X) for model in models])
        print('guessed_sales: {}'.format(guessed_sales))
        mean_sales = guessed_sales.mean(axis=0)
        relative_err = np.absolute((y - mean_sales) / y)
        result = np.sum(relative_err) / len(y)
        return result

    def evaluate_models_nostore_id(self, models, X, y):
        guessed_sales = np.array([model.guess(X) for model in models])
        print('ground_true: {} , guessed_sales: {}, diff: {}'.format
              (y, guessed_sales, (y-guessed_sales)))
        mean_sales = guessed_sales.mean(axis=0)
        relative_err = np.absolute((y - mean_sales) / y)
        result = np.sum(relative_err)
        return result

# inference = Inference()
# inference.inference_val_data()
# inference.inference_no_storeid()


