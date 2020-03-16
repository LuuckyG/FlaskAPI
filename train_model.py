import os
import time
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from pickle import load, dump

# Neural Net Preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.callbacks import EarlyStopping

from model import build_model, generate_text

# Import the data
testing = False

if testing:
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='cp1252')
else:
    dataset = pd.read_excel("database.xlsx")
    subject = 'AANLEIDING'
    text = dataset[dataset['Zwaartepunt'] == 'Programmatuur'][subject].values

vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

max_words = 50000  # Max size of the dictionary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
print(sequences[:5])

# Flatten the list of lists resulting from the tokenization. This will reduce the list
# to one dimension, allowing us to apply the sliding window technique to predict the next word
text = [item for sublist in sequences for item in sublist]
vocab_size = len(tokenizer.word_index)

print('Vocabulary size in this corpus: ', vocab_size)

# Training on 19 words to predict the 20th
sentence_len = 20
pred_len = 1
train_len = sentence_len - pred_len
seq = []

# Sliding window to generate train data
for i in range(len(text) - sentence_len):
    seq.append(text[i:i + sentence_len])

# Reverse dictionary to decode tokenized sequences back to words
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Each row in seq is a 20 word long window. We append he first 19 words as the input to predict the 20th word
X = []
y = []
for i in seq:
    X.append(i[:train_len])
    y.append(i[-1])

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_val = np.asarray(X_train[-int(len(X_train) * 0.2):])  # Get last 20% of training data for validation
y_val =  np.asarray(y_train[-int(len(y_train) * 0.2):])
X_train =  np.asarray(X_train[:-int(len(X_train) * 0.2)])
y_train =  np.asarray(y_train[:-int(len(y_train) * 0.2)])
X_test =  np.asarray(X_test)
y_test =  np.asarray(y_test)

# Hyperparameters
learning_rate = 0.003
loss_type = 'ce'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = 'adam'
dropout=0.3
num_cells = 256
rnn_layers = 3
activation='relu'


# define model
model = build_model(num_cells=num_cells, 
                    rnn_layers=rnn_layers, 
                    vocab_size=vocab_size, 
                    train_len=train_len, 
                    dropout=dropout,
                    activation=activation,
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics)



time_now = datetime.now()
results_path = Path('./results')
results_directory = results_path / ''.join(loss_type + '_lr_' + str(learning_rate) + '_t_' + time_now.strftime('%H-%M'))
results_directory.mkdir(exist_ok=True, parents=True)

checkpoint = ModelCheckpoint(str(results_directory) + "/weights.hdf5", monitor='val_' + loss, 
                             verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
callbacks_list = [checkpoint]
 
# Tensorboard logger.
tensorboard = TensorBoard(log_dir=results_directory, write_graph=True, write_grads=False, write_images=True)

history = model.fit(X_train,
                    y_train,
                    epochs=200,
                    batch_size=128,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_data=(X_val, y_val))


# # Create a dictionary containing all parameters.
# parameters['training']['results_directory'] = str(results_directory)
# parameters['training']['checkpoint_metric'] = repr(checkpoint_metric)
# parameters['training']['checkpoint_mode'] = repr(checkpoint_mode)
# parameters['training']['metrics'] = repr(metrics)

# with open(str(results_directory / 'parameters.json'), 'w') as json_file:
#     json.dump(parameters, fp=json_file, indent=4)

# Save tokenizer
pickle.dump(tokenizer, open(results_directory / 'tokenizer.pkl', 'wb'))
joblib.dump(model, str(results_directory) + '/{}_model.pkl'.format(subject))
model.save(str(results_directory / 'final_model.h5'))

# Generate text
test_string = 'De aanvrager'
sequence_length = 15
generate_text(model=model, seq=test_string, max_words=max_words, max_len=sequence_length)
