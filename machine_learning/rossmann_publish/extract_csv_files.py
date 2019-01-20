import pickle
import csv

class InputFile():
    """
    Usage:
    InputFile()
    """

    def __init__(self):
        train_data = "./data/train.csv"
        store_data = "./data/store.csv"
        store_states = './data/store_states.csv'

        TRAIN_OUTPUT = "./data/train_data.pickle"
        STORE_DATA_OUTPUT = "./data/store_data.pickle"

        with open(train_data) as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            with open(TRAIN_OUTPUT, 'wb') as f:
                data = self.csv2dicts(data)
                data = data[::-1]
                pickle.dump(data, f, -1)
                print(data[:3])

        with open(store_data) as csvfile, open(store_states) as csvfile2:
            data = csv.reader(csvfile, delimiter=',')
            state_data = csv.reader(csvfile2, delimiter=',')
            with open(STORE_DATA_OUTPUT, 'wb') as f:
                data = self.csv2dicts(data)
                state_data = self.csv2dicts(state_data)
                self.set_nan_as_string(data)
                for index, val in enumerate(data):
                    state = state_data[index]
                    val['State'] = state['State']
                    data[index] = val
                pickle.dump(data, f, -1)
                print(data[:2])

    def csv2dicts(self, csvfile):
        data = []
        keys = []
        for row_index, row in enumerate(csvfile):
            if row_index == 0:
                keys = row
                print(row)
                continue
            # if row_index % 10000 == 0:
            #     print(row_index)
            data.append({key: value for key, value in zip(keys, row)})
        return data


    def set_nan_as_string(self, data, replace_str='0'):
        for i, x in enumerate(data):
            for key, value in x.items():
                if value == '':
                    x[key] = replace_str
            data[i] = x

InputFile()


