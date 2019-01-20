import pickle
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import random
random.seed(42)

class SupervisedFeature():
    def __init__(self):
        TRAIN_DATA_OUTPUT = './data/train_data.pickle'
        STORE_DATA_OUTPUT = './data/store_data.pickle'

        with open(TRAIN_DATA_OUTPUT, 'rb') as f:
            train_data = pickle.load(f)
            num_records = len(train_data)
        with open(STORE_DATA_OUTPUT, 'rb') as f:
            self.store_data = pickle.load(f)

        train_data_X = []
        train_data_y = []

        for record in train_data:
            if record['Sales'] != '0' and record['Open'] != '':
                fl = self.feature_list(record)
                train_data_X.append(fl)
                train_data_y.append(int(record['Sales']))
        print("Number of train datapoints: ", len(train_data_y))

        print(min(train_data_y), max(train_data_y))

        full_X = train_data_X
        full_X = np.array(full_X)
        train_data_X = np.array(train_data_X)
        les = []
        for i in range(train_data_X.shape[1]):
            le = preprocessing.LabelEncoder()
            le.fit(full_X[:, i])
            les.append(le)
            train_data_X[:, i] = le.transform(train_data_X[:, i])


        LABEL_ENCODINGS = './data/les.pickle'
        FEATURE_TRAIN_DATA = './data/feature_train_data.pickle'

        with open(LABEL_ENCODINGS, 'wb') as f:
            pickle.dump(les, f, -1)

        train_data_X = train_data_X.astype(int)
        train_data_y = np.array(train_data_y)

        with open(FEATURE_TRAIN_DATA, 'wb') as f:
            pickle.dump((train_data_X, train_data_y), f, -1)
            print(train_data_X[0], train_data_y[0])

    def feature_list(self, record):
        dt = datetime.strptime(record['Date'], '%Y-%m-%d')
        store_index = int(record['Store'])
        year = dt.year
        month = dt.month
        day = dt.day
        day_of_week = int(record['DayOfWeek'])
        try:
            store_open = int(record['Open'])
        except:
            store_open = 1

        promo = int(record['Promo'])

        return [store_open,
                store_index,
                day_of_week,
                promo,
                year,
                month,
                day,
                self.store_data[store_index - 1]['State']
                ]


SupervisedFeature()