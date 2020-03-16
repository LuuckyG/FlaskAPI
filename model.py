import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

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

# Import the data
dataset = pd.read_excel("database.xlsx")
subject = 'AANLEIDING'
text = dataset[dataset['Zwaartepunt'] == 'Programmatuur'][subject].values


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='cp1252')
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

max_words = 50000  # Max size of the dictionary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(dataset[subject].values)
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


# define model
model = Sequential([
    Embedding(vocab_size + 1, 25, input_length=train_len),
    LSTM(256, return_sequences=True),
    LSTM(256),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(vocab_size + 1, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filepath = "./model_2_weights.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
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

# Save tokenizer
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
joblib.dump(model, '{}_model.pkl'.format(subject))
model.save('model_weights.hdf5')


def gen(model, seq, max_len=20):
    """ Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len"""
    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    max_len = max_len + len(tokenized_sent[0])
    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. the `pad_sequences` function adds
    # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
    while len(tokenized_sent[0]) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-19:], maxlen=19)
        op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
        tokenized_sent[0].append(op.argmax() + 1)

    return " ".join(map(lambda x: reverse_word_map[x], tokenized_sent[0]))


test_string = 'De aanvrager'
sequence_length = 15
gen(model, test_string, sequence_length)
