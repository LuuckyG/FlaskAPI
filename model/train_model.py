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
from .models import build_model, generate_text
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
    model = build_model(num_cells=cfg['num_cells'], 
                        rnn_layers=cfg['rnn_layers'], 
                        vocab_size=vocab_size, 
                        train_len=cfg['sentence_length'] - 1, 
                        dropout=cfg['dropout'],
                        activation=cfg['activation'],
                        optimizer=optimizer,
                        loss=loss,
                        metrics=cfg['metrics'])

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
                        verbose=1,
                        validation_data=(X_val, y_val))    

    cfg = save_config(model, tokenizer, cfg, reverse_word_map, results_directory)
    return model, cfg


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
