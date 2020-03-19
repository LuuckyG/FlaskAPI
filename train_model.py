import os
import json
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
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

from model import build_model, generate_text
from hparams import HParams


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
    
    vocab = sorted(set(inputs))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in inputs])

    tokenizer = Tokenizer(num_words=cfg['max_words'])
    tokenizer.fit_on_texts(inputs)
    sequences = tokenizer.texts_to_sequences(inputs)
    print(sequences[:5])

    # Flatten the list of lists resulting from the tokenization. This will reduce the list
    # to one dimension, allowing us to apply the sliding window technique to predict the next word
    text = [item for sublist in sequences for item in sublist]
    vocab_size = len(tokenizer.word_index)

    print('Vocabulary size in this corpus: ', vocab_size)

    # Training on n-1 words to predict the nth
    sentence_len = cfg['sentence_length']
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=cfg['test_size'], 
                                                        random_state=np.random.RandomState(cfg['seed']))

    X_val = np.asarray(X_train[-int(len(X_train) * 0.2):])  # Get last 20% of training data for validation
    y_val =  np.asarray(y_train[-int(len(y_train) * 0.2):])
    X_train =  np.asarray(X_train[:-int(len(X_train) * 0.2)])
    y_train =  np.asarray(y_train[:-int(len(y_train) * 0.2)])
    X_test =  np.asarray(X_test)
    y_test =  np.asarray(y_test)

    # Model parameters.
    if cfg['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=cfg['learning_rate'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(learning_rate=cfg['learning_rate'])
    else:
        raise TypeError("Specified optimizer type does not exist.")

    if cfg['loss_type'] == 'ce':
        loss = 'categorical_crossentropy'
        checkpoint_metric = 'val_' + loss
    else:
        raise TypeError("Unrecognized loss type.")

    # define model
    model = build_model(num_cells=cfg['num_cells'], 
                        rnn_layers=cfg['rnn_layers'], 
                        vocab_size=vocab_size, 
                        train_len=train_len, 
                        dropout=cfg['dropout'],
                        activation=cfg['activation'],
                        optimizer=optimizer,
                        loss=loss,
                        metrics=cfg['metrics'])

    # Set up saving folder
    time_now = datetime.now()
    results_directory = results_path / ''.join(cfg['loss_type'] + '_lr_' 
                        + str(cfg['learning_rate']) + '_' + time_now.strftime('%H-%M'))
    results_directory.mkdir(exist_ok=True, parents=True)

    model_early_stop_filename = results_directory / 'model_checkpoint.h5'
    checkpoint_metric = 'val_' + loss
    checkpoint_mode = 'min'
    checkpoint = ModelCheckpoint(str(model_early_stop_filename), monitor=checkpoint_metric, 
                                verbose=1, save_best_only=True, save_weights_only=False, mode=checkpoint_mode, period=1)
    
    # Model logger.
    csv_logger = CSVLogger(str(results_directory / 'training.log'))

    # Tensorboard logger.
    tensorboard = TensorBoard(log_dir=results_directory, write_graph=True, write_grads=False, write_images=True)

    callbacks_list = [checkpoint, csv_logger, tensorboard]

    # Train model
    history = model.fit(X_train,
                        y_train,
                        epochs=cfg['epochs'],
                        batch_size=cfg['batch_size'],
                        callbacks=callbacks_list,
                        verbose=1,
                        validation_data=(X_val, y_val))    

    # Save training configuration
    cfg['results_directory'] = str(results_directory)
    cfg['results_path'] = str(cfg['results_path'])
    cfg['path_to_file'] = str(cfg['path_to_file'])
    cfg['model_path'] = str(cfg['model_path'])
    cfg['checkpoint_metric'] = repr(checkpoint_metric)
    cfg['checkpoint_mode'] = repr(checkpoint_mode)
    cfg['reverse_word_map'] = reverse_word_map

    with open(str(results_directory / 'config.json'), 'w') as json_file:
        json.dump(cfg, fp=json_file, indent=4)

    # Serialize model to JSON
    model_json = model.to_json()
    with open(str(results_directory / 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    
    # Save model
    pickle.dump(tokenizer, open(results_directory / 'tokenizer.pkl', 'wb'))
    model.save(str(results_directory / 'final_model.h5'))

    return model, cfg

if __name__ == "__main__":
    parameters = HParams().args

    model, config = train_model(results_path=parameters.results_path,
                                path_to_file=parameters.path_to_file,
                                cfg=parameters.__dict__)
    
    if parameters.generate:
        gen_text = generate_text(model=model, 
                                 seq=parameters.input_string,
                                 reverse_word_map = config['reverse_word_map'],
                                 max_words=parameters.max_words, 
                                 max_len=parameters.sequence_length
                                 )
        
        with open(config['results_directory'] + '/gen_text.txt', 'w') as f:
            f.write(gen_text)
