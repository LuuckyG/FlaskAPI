import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime

# Neural Net Preprocessing
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing import sequence

# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, Callback

# Own scripts
from hparams import HParams
from .models import build_simple_model, generate_text
from utils.preprocess import split_data
from utils.logging import checkpoint_log, save_config
from .utils import textgenrnn_encode_cat
from .textgenrnn import textgenrnn
from .textgenrnn2 import TextGeneration


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
        # dataset = pd.read_excel(path_to_file)
        # inputs = dataset[dataset['Zwaartepunt'] == 'Programmatuur'][cfg['subject']].values
        with open('aanleiding.txt', 'r', encoding='utf8', errors='ignore') as f:
            texts = [f.read()]

    
    # Tokenize inputs
    tokenizer = Tokenizer(filters='', 
                          lower=cfg['word_level'],
                          char_level=cfg['word_level'])
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit vocab to max_words
    max_words = cfg['max_words']
    reverse_word_map = {k: v for (k, v) in tokenizer.word_index.items() if v <= max_words}

    if cfg.get('single_text', False): 
        tokenizer.word_index[META_TOKEN] = len(tokenizer.word_index) + 1

    # Create vocabulary
    vocab = tokenizer.word_index
    vocab_size = len(tokenizer.word_index)
    indices_char = dict((vocab[c], c) for c in vocab)

    print('Vocabulary size in this corpus: ', vocab_size)

    # if cfg['word_level']:
    #     # If training word level, must add spaces around each
    #     # punctuation. https://stackoverflow.com/a/3645946/9314418
    #     punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—…'
    #     for i in range(len(texts)):
    #         texts[i] = re.sub('([{}])'.format(punct), r' \1 ', texts[i])
    #         texts[i] = re.sub(' {2,}', ' ', texts[i])
    #     texts = [text_to_word_sequence(text, filters='') for text in texts]
    
    # # Create training data
    # X_train, X_val, y_train, y_val = split_data(texts, vocab_size, cfg)

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

    # Set up results folder
    time_now = datetime.now()
    results_dir = results_path / ''.join(cfg['loss_type'] + '_lr_' 
                    + str(cfg['learning_rate']) + '_' + time_now.strftime('%H-%M'))
    results_dir.mkdir(exist_ok=True, parents=True)
    cfg['results_dir'] = str(results_dir)

    # Callback list contains ModelCheckpoint, CSVlogger and Tensorboard
    callbacks_list = checkpoint_log(results_dir, loss)

    if cfg['text_genrnn']:
        # Original code from Max Woolf (Github: Minimaxir):
        # https://github.com/minimaxir/textgenrnn

        model = textgenrnn(model_folder=str(results_dir))

        train_function = model.train_from_file if cfg['line_delimited'] else model.train_from_largetext_file

        train_function(
            file_path='aanleiding.txt',
            new_model=True,
            num_epochs=cfg['epochs'],
            gen_epochs=cfg['gen_epochs'],
            batch_size=cfg['batch_size'],
            train_size=1-cfg['test_size'],
            dropout=cfg['dropout'],
            validation=cfg['validation'],
            is_csv=cfg['is_csv'],
            rnn_layers=cfg['rnn_layers'],
            rnn_size=cfg['rnn_size'],
            rnn_bidirectional=cfg['rnn_bidirectional'],
            max_length=cfg['sentence_length'],
            dim_embeddings=100,
            word_level=cfg['word_level'])           

    else:
        # Define model
        model = build_simple_model(
            rnn_size=cfg['rnn_size'], 
            rnn_layers=cfg['rnn_layers'], 
            vocab_size=vocab_size, 
            train_len=cfg['sentence_length'] - 1, 
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            optimizer=optimizer,
            loss=loss,
            metrics=cfg['metrics'])

        # Train model
        history = model.fit(X_train,
                            y_train,
                            epochs=cfg['epochs'],
                            batch_size=cfg['batch_size'],
                            callbacks=callbacks_list,
                            verbose=2,
                            validation_data=(X_val, y_val))

        cfg = save_config(model, tokenizer, cfg, reverse_word_map, results_directory)           # Wont work yet with textgenrnn

    return model, cfg


if __name__ == "__main__":
    parameters = HParams().args

    model, config = train_model(results_path=parameters.results_path,
                                path_to_file=parameters.path_to_file,
                                cfg=parameters.__dict__)
    
    if parameters.generate:
        if parameters.text_genrnn:  
            if config['line_delimited']:
                n = 1000
                max_gen_length = 60 if config['word_level'] else 300
            else:
                n = 1
                max_gen_length = 2000 if config['word_level'] else 1000
                
            gen_file = '{}/gen_text.txt'.format(config['results_dir'])
            temperature =[float(t) for t in config['temperature']]

            model.generate_to_file(gen_file,
                                    temperature=temperature,
                                    prefix=config['input_string'],
                                    n=n,
                                    max_gen_length=max_gen_length)
        else:
            with open(config['tokenizer_path'], 'rb') as handle:
                tokenizer = pickle.load(handle)

            gen_text = generate_text(model=model,
                                    tokenizer=tokenizer,
                                    inputs=parameters.input_string,
                                    reverse_word_map=config['reverse_word_map'],
                                    train_len=config['sentence_length'] - 1,
                                    max_len=parameters.sequence_length
                                    )
            
            with open(config['results_dir'] + '/gen_text.txt', 'w') as f:
                f.write(gen_text)
