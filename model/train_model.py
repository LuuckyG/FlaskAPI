import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime

from hparams import HParams
from .utils import textgenrnn_encode_cat
from .textgenrnn import textgenrnn


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
        texts = dataset[dataset['Zwaartepunt'] == 'Programmatuur'][cfg['subject']].values

    # Set up results folder
    time_now = datetime.now()    
    results_dir = results_path / (('word' if cfg['word_level'] else 'char') 
                                + ('_bidir' if cfg['rnn_bidirectional'] else '') 
                                + '_l' + str(cfg['sentence_length']) 
                                + '_d' + str(cfg['rnn_layers']) 
                                + '_w' + str(cfg['rnn_size']) 
                                + '_' + time_now.strftime('%H-%M'))
    results_dir.mkdir(exist_ok=True, parents=True)
    cfg['results_dir'] = str(results_dir)

    # Original code from Max Woolf (Github: Minimaxir):
    # https://github.com/minimaxir/textgenrnn
    model = textgenrnn(model_folder=str(results_dir))
    cfg['name'] = model.config['name']

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

    return model, cfg


if __name__ == "__main__":
    parameters = HParams().args

    model, config = train_model(results_path=parameters.results_path,
                                path_to_file=parameters.path_to_file,
                                cfg=parameters.__dict__)

    if parameters.generate:
        if config['line_delimited']:
            n = 1000
            max_gen_length = 60 if config['word_level'] else 1500
        else:
            n = 1
            max_gen_length = 1000 if config['word_level'] else 1500
            
        gen_file = '{}/gen_text.txt'.format(config['results_dir'])
        temperature =[float(t) for t in config['temperature']]

        model.generate_to_file(gen_file,
                                temperature=temperature,
                                prefix=config['input_string'],
                                n=n,
                                max_gen_length=max_gen_length)
        