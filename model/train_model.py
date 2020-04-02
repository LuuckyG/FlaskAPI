import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime

# Neural Net Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras import optimizers

from hparams import HParams
from .models import build_simple_model, generate_text
from utils.preprocess import split_data
from utils.logging import checkpoint_log, save_config


def train_model(results_path: Path, path_to_file: str, cfg: dict):
    """
        Function to train a LSTM model for text generation.
        Args:
            results_path: Folder where subfolder with results of training and model are saved.
            path_to_file: Path to where input data is located.
            cfg: Dictionary with all hyperparameter, model and text generation settings are stored.
        
        Returns:
            model: Trained model.
    """

    # Import data
    if cfg['testing']:
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        inputs = open(path_to_file, 'rb').read().decode(encoding='cp1252')
    else:
        dataset = pd.read_excel(path_to_file)
        inputs = dataset[dataset['Zwaartepunt'] == 'Programmatuur'][cfg['subject']].values
    
    # Tokenize inputs
    tokenizer = Tokenizer(num_words=cfg['max_words'])
    tokenizer.fit_on_texts(inputs)
    sequences = tokenizer.texts_to_sequences(inputs)
    
    # Reverse dictionary to decode tokenized sequences back to words
    reverse_word_map = {str(k): v for k, v in map(reversed, tokenizer.word_index.items())}

    # Flatten the list of lists resulting from the tokenization. This will reduce the list
    # to one dimension, allowing us to apply the sliding window technique to predict the next word
    text = [item for sublist in sequences for item in sublist]

    # Create vocab
    vocab = sorted(set(inputs))
    vocab_size = len(tokenizer.word_index)

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in inputs])

    print('Vocabulary size in this corpus: ', vocab_size)

    # Create training data
    X_train, X_val, y_train, y_val = split_data(text, vocab_size, cfg)

    # Model parameters.
    if cfg['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=cfg['learning_rate'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(learning_rate=cfg['learning_rate'])
    else:
        raise TypeError("Specified optimizer type does not exist.")

    if cfg['loss_type'] == 'ce':
        loss = 'categorical_crossentropy'
    else:
        raise TypeError("Unrecognized loss type.")

    # define model
    model = build_simple_model(
                num_cells=cfg['num_cells'], 
                rnn_layers=cfg['rnn_layers'], 
                vocab_size=vocab_size, 
                train_len=cfg['sentence_length'] - 1, 
                dropout=cfg['dropout'],
                activation=cfg['activation'],
                optimizer=optimizer,
                loss=loss,
                metrics=cfg['metrics'])

    if text_genrnn:
        textgen = textgenrnn(name=model_name)

        train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file

        train_function(
            file_path=file_name,
            new_model=True,
            num_epochs=train_cfg['num_epochs'],
            gen_epochs=train_cfg['gen_epochs'],
            batch_size=train_cfg['batch_size'],
            train_size=train_cfg['train_size'],
            dropout=train_cfg['dropout'],
            validation=train_cfg['validation'],
            is_csv=train_cfg['is_csv'],
            rnn_layers=model_cfg['rnn_layers'],
            rnn_size=model_cfg['rnn_size'],
            rnn_bidirectional=model_cfg['rnn_bidirectional'],
            max_length=model_cfg['max_length'],
            dim_embeddings=100,
            word_level=model_cfg['word_level'])

        if train_cfg['line_delimited']:
            n = 1000
            max_gen_length = 60 if model_cfg['word_level'] else 300
        else:
            n = 1
            max_gen_length = 2000 if model_cfg['word_level'] else 1000
            
            timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
            gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)

            textgen.generate_to_file(gen_file,
                                    temperature=temperature,
                                    prefix=prefix,
                                    n=n,
                                    max_gen_length=max_gen_length)

    # Set up results folder
    time_now = datetime.now()
    results_directory = results_path / ''.join(cfg['loss_type'] + '_lr_' 
                        + str(cfg['learning_rate']) + '_' + time_now.strftime('%H-%M'))
    results_directory.mkdir(exist_ok=True, parents=True)

    # Callback list contains ModelCheckpoint, CSVlogger and Tensorboard
    callbacks_list = checkpoint_log(results_directory, loss)

    # Train model
    history = model.fit(X_train,
                        y_train,
                        epochs=cfg['epochs'],
                        batch_size=cfg['batch_size'],
                        callbacks=callbacks_list,
                        verbose=2,
                        validation_data=(X_val, y_val))

    cfg = save_config(model, tokenizer, cfg, reverse_word_map, results_directory)
    return model, cfg


#####################################
# Original name = model_training.py

  
import numpy as np

from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

from .utils import textgenrnn_encode_cat


def generate_sequences_from_texts(texts, indices_list,
                                  textgenrnn, context_labels,
                                  batch_size=128):
    is_words = textgenrnn.config['word_level']
    is_single = textgenrnn.config['single_text']
    max_length = textgenrnn.config['max_length']
    meta_token = textgenrnn.META_TOKEN

    if is_words:
        new_tokenizer = Tokenizer(filters='', char_level=True)
        new_tokenizer.word_index = textgenrnn.vocab
    else:
        new_tokenizer = textgenrnn.tokenizer

    while True:
        np.random.shuffle(indices_list)

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for row in range(indices_list.shape[0]):
            text_index = indices_list[row, 0]
            end_index = indices_list[row, 1]

            text = texts[text_index]

            if not is_single:
                text = [meta_token] + list(text) + [meta_token]

            if end_index > max_length:
                x = text[end_index - max_length: end_index + 1]
            else:
                x = text[0: end_index + 1]
            y = text[end_index + 1]

            if y in textgenrnn.vocab:
                x = process_sequence([x], textgenrnn, new_tokenizer)
                y = textgenrnn_encode_cat([y], textgenrnn.vocab)

                X_batch.append(x)
                Y_batch.append(y)

                if context_labels is not None:
                    context_batch.append(context_labels[text_index])

                count_batch += 1

                if count_batch % batch_size == 0:
                    X_batch = np.squeeze(np.array(X_batch))
                    Y_batch = np.squeeze(np.array(Y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    # print(X_batch.shape)

                    if context_labels is not None:
                        yield ([X_batch, context_batch], [Y_batch, Y_batch])
                    else:
                        yield (X_batch, Y_batch)
                    X_batch = []
                    Y_batch = []
                    context_batch = []
                    count_batch = 0


def process_sequence(X, textgenrnn, new_tokenizer):
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=textgenrnn.config['max_length'])

    return X



#################################

if __name__ == "__main__":
    parameters = HParams().args

    model, config = train_model(results_path=parameters.results_path,
                                path_to_file=parameters.path_to_file,
                                cfg=parameters.__dict__)
    
    if parameters.generate:
        with open(config['tokenizer_path'], 'rb') as handle:
            tokenizer = pickle.load(handle)

        gen_text = generate_text(model=model,
                                 tokenizer=tokenizer,
                                 inputs=parameters.input_string,
                                 reverse_word_map=config['reverse_word_map'],
                                 train_len=config['sentence_length'] - 1,
                                 max_len=parameters.sequence_length
                                 )
        
        with open(config['results_directory'] + '/gen_text.txt', 'w') as f:
            f.write(gen_text)
