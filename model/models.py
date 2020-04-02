import numpy as np 

from tensorflow.keras import optimizers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_simple_model(num_cells, 
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
                  train_len=20,
                  max_len=100, 
                  **kwargs):
    """ 
    Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len

    Args:
        model: A pre-trained tensorflow.keras model
        tokenizer: Tokenizer object
        inputs: Input text for the model to start prediction on.
        reverse_word_map: Reverse dictionary to decode tokenized sequences back to words
        train_len: Lenght of sentence used to train the model
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
        padded_sentence = pad_sequences(tokenized_sent[-train_len:], maxlen=train_len)
        op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
        tokenized_sent[0].append(op.argmax() + 1)

    return " ".join(map(lambda x: reverse_word_map[str(x)], tokenized_sent[0]))


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import config as config
from .AttentionWeightedAverage import AttentionWeightedAverage


def textgenrnn_model(num_classes, cfg, context_size=None,
                     weights_path=None,
                     dropout=0.0,
                     optimizer=Adam(lr=4e-3)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    input = Input(shape=(cfg['max_length'],), name='input')
    embedded = Embedding(num_classes, cfg['dim_embeddings'],
                         input_length=cfg['max_length'],
                         name='embedding')(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i == 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(cfg, i + 1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = AttentionWeightedAverage(name='attention')(seq_concat)
    output = Dense(num_classes, name='output', activation='softmax')(attention)

    if context_size is None:
        model = Model(inputs=[input], outputs=[output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    else:
        context_input = Input(
            shape=(context_size,), name='context_input')
        context_reshape = Reshape((context_size,),
                                  name='context_reshape')(context_input)
        merged = concatenate([attention, context_reshape], name='concat')
        main_output = Dense(num_classes, name='context_output',
                            activation='softmax')(merged)

        model = Model(inputs=[input, context_input],
                      outputs=[main_output, output])
        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      loss_weights=[0.8, 0.2])

    return model


'''
Create a new LSTM layer per parameters. Unfortunately,
each combination of parameters must be hardcoded.
The normal LSTMs use sigmoid recurrent activations
for parity with CuDNNLSTM:
https://github.com/keras-team/keras/issues/8860
'''

'''
FIXME
From TensorFlow 2 you do not need to specify CuDNNLSTM.
You can just use LSTM with no activation function and it will
automatically use the CuDNN version.
This part can probably be cleaned up.
'''


def new_rnn(cfg, layer_num):
    use_cudnnlstm = K.backend() == 'tensorflow'  # and len(config.get_visible_devices('GPU')) > 0
    if use_cudnnlstm:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    name='rnn_{}'.format(layer_num))
    else:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(layer_num))
