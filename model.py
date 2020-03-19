import numpy as np 

from tensorflow.keras import optimizers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_model(num_cells, 
                rnn_layers, 
                vocab_size, 
                train_len, 
                dropout=0.3,
                activation='relu',
                optimizer=optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                **kwargs):

    """Function to build and return a recurrent LSTM-based model"""

    # Define model
    model = Sequential()
    model.add(Embedding(vocab_size + 1, 25, input_length=train_len))

    for n in range(rnn_layers):
        if n != (rnn_layers - 1):
            model.add(LSTM(num_cells, return_sequences=True))
            model.add(Dropout(dropout))
        else:
            # Last LSTM layer
            model.add(LSTM(num_cells))
    
    model.add(Dense(num_cells, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size + 1, activation='softmax'))

    model.summary()

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    return model


def generate_text(model, 
                  tokenizer, 
                  inputs, 
                  reverse_word_map, 
                  max_len=20, 
                  **kwargs):
    """ 
    Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len

    Args:
        model: A pre-trained tensorflow.keras model
        tokenizer: Tokenizer object
        inputs: Input text for the model to start prediction on.
        reverse_word_map: Reverse dictionary to decode tokenized sequences back to words
        max_len: Number of words of generated text.

    Returns:
        A generated text.
    """
    
    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([inputs])
    max_len = max_len + len(tokenized_sent[0])

    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. 
    while len(tokenized_sent[0]) < max_len:
        # Pad on left of sentence
        padded_sentence = pad_sequences(tokenized_sent[-(max_len-1):], maxlen=maxlen-1)
        op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
        tokenized_sent[0].append(op.argmax() + 1)

    return " ".join(map(lambda x: reverse_word_map[x], tokenized_sent[0]))
