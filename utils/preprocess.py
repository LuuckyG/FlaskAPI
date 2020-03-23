import numpy as np 
import pandas as pd

from pathlib import Path
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def split_data(text: list, vocab_size: int, cfg: dict):
    """
    Function to split input text into training, validation and test datasets and save them to data folder.

    Args:
        text: List with full text dataset that needs to be split.
        vocab_size: Size of vocabulary.
        cfg: Dictionary containing information over 'sentence_length', 'test_size' and 'seed' hyperparameters.
    
    Returns:
        Training and validation data and labels in numpy ndarray format.
    """
    
    # Training on n-1 words to predict the nth
    sentence_len = cfg['sentence_length']
    train_len = sentence_len - 1
    seq = []

    # Sliding window to generate train data
    for i in range(len(text) - sentence_len):
        seq.append(text[i:i + sentence_len])

    # Each row in seq is a 20 word long window. We append he first 19 words as the input to predict the 20th word
    X = []
    y = []
    for i in seq:
        X.append(i[:train_len])
        y.append(i[-1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=cfg['test_size'], 
                                                        random_state=np.random.RandomState(cfg['seed']))

    # Get last 20% of training data for validation
    X_val = np.asarray(X_train[-int(len(X_train) * 0.2):])
    X_train =  np.asarray(X_train[:-int(len(X_train) * 0.2)])
    X_test =  np.asarray(X_test)

    y_val =  np.asarray(y_train[-int(len(y_train) * 0.2):]).reshape((-1,1))
    y_train =  np.asarray(y_train[:-int(len(y_train) * 0.2)]).reshape((-1,1))
    y_test =  np.asarray(y_test).reshape((-1,1))

    # Save data and labels together in one file
    header = list(range(0, train_len)) + ['label']
    save_data(data=np.concatenate((X_train, y_train), axis=1), location='data/train.csv', header=header)
    save_data(data=np.concatenate((X_val, y_val), axis=1), location='data/val.csv', header=header)
    save_data(data=np.concatenate((X_test, y_test), axis=1), location='data/test.csv', header=header)

    return X_train, X_val, to_categorical(y_train, num_classes=vocab_size + 1), to_categorical(y_val, num_classes=vocab_size + 1)


def save_data(data: np.ndarray, location: str, header=None):
    """
    Function that saves input data to specified location.
    """
    Path(location).parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(data).to_csv(location, sep=',', header=header, index=None)
