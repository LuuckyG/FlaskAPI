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
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']):

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

    # Print summary
    model.summary()

    # Build model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    return model


def generate_text(model, seq, max_words=50000, max_len=20):
    """ Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len"""
    # Tokenize the input string
    tokenizer = Tokenizer(num_words=max_words)
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
