import numpy
numpy.random.seed(123)

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.models import model_from_json


class Model_Util():
    def __init__(self):
        pass

    def split_features(self, X):
        X_list = []

        store_index = X[..., [1]]
        X_list.append(store_index)

        day_of_week = X[..., [2]]
        X_list.append(day_of_week)

        promo = X[..., [3]]
        X_list.append(promo)

        year = X[..., [4]]
        X_list.append(year)

        month = X[..., [5]]
        X_list.append(month)

        day = X[..., [6]]
        X_list.append(day)

        State = X[..., [7]]
        X_list.append(State)

        return X_list


class Model(object):

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result



class NN_with_EntityEmbedding(Model):

    def __init__(self, X_train, y_train, X_val, y_val, Model_Util):
        self.epochs = 1
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.Model_Util = Model_Util
        self.fit(X_train, y_train, X_val, y_val)


    def preprocessing(self, X):
        X_list = self.Model_Util.split_features(X)
        return X_list

    def __build_keras_model(self):
        input_store = Input(shape=(1,))
        output_store = Embedding(1115, 10, name='store_embedding')(input_store)
        output_store = Reshape(target_shape=(10,))(output_store)

        input_dow = Input(shape=(1,))
        output_dow = Embedding(7, 6, name='dow_embedding')(input_dow)
        output_dow = Reshape(target_shape=(6,))(output_dow)

        input_promo = Input(shape=(1,))
        output_promo = Dense(1)(input_promo)

        input_year = Input(shape=(1,))
        output_year = Embedding(3, 2, name='year_embedding')(input_year)
        output_year = Reshape(target_shape=(2,))(output_year)

        input_month = Input(shape=(1,))
        output_month = Embedding(12, 6, name='month_embedding')(input_month)
        output_month = Reshape(target_shape=(6,))(output_month)

        input_day = Input(shape=(1,))
        output_day = Embedding(31, 10, name='day_embedding')(input_day)
        output_day = Reshape(target_shape=(10,))(output_day)

        input_germanstate = Input(shape=(1,))
        output_germanstate = Embedding(12, 6, name='state_embedding')(input_germanstate)
        output_germanstate = Reshape(target_shape=(6,))(output_germanstate)

        input_model = [input_store, input_dow, input_promo,
                       input_year, input_month, input_day, input_germanstate]

        output_embeddings = [output_store, output_dow, output_promo,
                             output_year, output_month, output_day, output_germanstate]

        output_model = Concatenate()(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       )
        self.save_model()
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def save_model(self):
        SAVED_MODEL_JSON = "./weight/saved_model_json.json"
        SAVED_MODEL_WEIGHT = "./weight/saved_model_weight.h5"

        model_json = self.model.to_json()

        with open(SAVED_MODEL_JSON, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(SAVED_MODEL_WEIGHT)
        print("Saved model to disk")

    def load_model(self, model_path, weight_path):
        model_file_path = model_path
        model_weight_path = weight_path
        json_file = open(model_file_path)
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weight_path)
        loaded_model.compile(loss='mean_absolute_error', optimizer='adam')

        return loaded_model

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)

class NN_with_EntityEmbedding_Loading(Model):

    def __init__(self, model_path, weight_path, y_train, y_val, Model_Util):
        super().__init__()

        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.Model_Util = Model_Util
        self.load_model(model_path, weight_path)


    def preprocessing(self, X):
        X_list = self.Model_Util.split_features(X)
        return X_list

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)


    def load_model(self, model_path, weight_path):
        model_file_path = model_path
        model_weight_path = weight_path
        json_file = open(model_file_path)
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weight_path)
        loaded_model.compile(loss='mean_absolute_error', optimizer='adam')
        self.model = loaded_model

        return loaded_model

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
