import time
start_time = time.time()
import os, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "8"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


         
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')            
            
#test = pd.read_csv("../input/test_1000obs.csv")
test = pd.read_csv("../input/test.csv")
# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embedding_path = "../input/glove.840B.300d.txt"
file_path = "../result/best_model_m100000.hdf5"
#submission_file_path = "../input/sample_submission_1000obs.csv"
submission_file_path = "../input/sample_submission.csv"
target_file_path = "../result/submission_gru_conv.csv"

embed_size = 300
max_features = 100000
max_len = 150

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
test["comment_text"].fillna("no comment")


raw_text_test = test["comment_text"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)

tk.fit_on_texts(raw_text_test)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

test = pad_sequences(test.comment_seq, maxlen = max_len)

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    

model = load_model(file_path)
#model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

model.load_weights(file_path)     


pred = model.predict(test, batch_size = 4096, verbose = 1)


submission = pd.read_csv(submission_file_path)
submission[list_classes] = (pred)
submission.to_csv(target_file_path, index = False)

# Measure duration
end_time = time.time()
duration = end_time - start_time
print ('Start time: ' + time.ctime(start_time))
print ('End time: ' + time.ctime(end_time))
print ('Total execution time: ' + str(duration/60))

