# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

"""
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
"""

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
start_time = time.time()


##################################################
# Hyperparameters
##################################################
max_features = 20000
maxlen = 300
batch_size = 32
epochs = 2

##################################################
# Process Input file
##################################################
#train = pd.read_csv("../input/train.csv")
#train = pd.read_csv("../input/train_1000obs.csv")
#print("train shape: ", train.describe())
#test = pd.read_csv("../input/small_test_file_40obs.csv")
test = pd.read_csv("../input/test.csv")
#print("test shape: ", test.describe())
#test = pd.read_csv("../input/test.csv")

# Extract samples by 100%
#train = train.sample(frac=1)
#print(train)

# Replace NA with "CVxTz"
#list_sentences_train = train["comment_text"].fillna("CVxTz").values
# print(type(list_sentences_train))
# print(list_sentences_train.shape)

# Extract label values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#y = train[list_classes].values
#print("y: ", y)

# Replace NA with "CVxTz"
list_sentences_test = test["comment_text"].fillna("CVxTz").values


##################################################
# Tokenize Input file
##################################################

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_test))

#print("word counts: ", tokenizer.word_counts)
#print("word index: ", tokenizer.word_index)
#print("word docs: ", tokenizer.word_docs)
#print("document_counts: ", tokenizer.document_count)

# Make a list of indices of words
#list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#print(type(list_tokenized_train))
#print(list_tokenized_train)
#print(len(list_tokenized_train))

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# Fill a padding until maxlen
#X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()

"""
print("model layers:\n", model.layers)
print("model inputs:\n", model.inputs)
print("model outputs:\n", model.outputs)
"""

file_path="weights_base.best.hdf5"
#checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


#callbacks_list = [checkpoint, early] #early

#print("X_t: \n", X_t.shape)
#print("y: \n", y.shape)

"""
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, 
          validation_split=0.1, callbacks=callbacks_list)
"""
model.load_weights(file_path)

y_test = model.predict(X_te)

print("Shape of y_test: ", np.shape(y_test))

sample_submission = pd.read_csv("../input/sample_submission.csv")

print("Shape of sample_submission: ", np.shape(sample_submission))

sample_submission[list_classes] = y_test

sample_submission.to_csv("../result/baseline_0301.csv", index=False)


# Measure duration
end_time = time.time()
duration = end_time - start_time
print ('Start time: ' + time.ctime(start_time))
print ('End time: ' + time.ctime(end_time))
print ('Total execution time: ' + str(duration/60))


