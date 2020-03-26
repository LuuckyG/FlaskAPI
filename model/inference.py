import json
import pickle
import argparse

from pathlib import Path
from tensorflow.keras.models import load_model

from .models import generate_text
from hparams import HParams


def float_input(s):
    return map(float, s.split(','))


def get_parser():
    arg_parser = argparse.ArgumentParser()

    # Required parameters
    arg_parser.add_argument('--model_path', '--mp', type=Path,
                            help="Path to pretrained model (hdf5 format).")
                            # C:/Users/luukg/Documents/01_Evolvalor/FlaskAPI/results/test
    arg_parser.add_argument('--results_path', '--rp', type=Path,
                            help="Folder where the generated text should be saved to.")
    arg_parser.add_argument('--inputs', '--i', nargs='+', type=str,
                            help="Provide input string for the model to predict the next words.")

    # Optional parameters
    arg_parser.add_argument('--sequence_length', type=int, default=100,
                            help="The length of the sentence generated.")

    return arg_parser.parse_args()


def inference(model_path, results_path, inputs, max_len):
    """
    Function to load in pre-trained model and generate text based on inputs.
    The generated text is saved in .txt file.

    Args:
        model_path: Path to pretrained model (hdf5 format).
        results_path: Folder where the generated text should be saved to.
        inputs: Provide input string for the model to predict the next words.
        max_len: The number of words of the sentence generated.
    
    Returns:
        Generated text.    
    """

    model = load_model(str(model_path / 'final_model.h5'))

    with open(model_path / 'config.json') as cfg:
        config = json.load(cfg)

    with open(model_path / 'tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    return generate_text(model=model, 
                         tokenizer=tokenizer, 
                         inputs=' '.join(inputs), 
                         reverse_word_map=config['reverse_word_map'], 
                         train_len=config['sentence_length'] - 1, 
                         max_len=max_len)


if __name__ == "__main__":
    params = get_parser()

    gen_text = inference(model_path=params.model_path,
                         results_path=params.results_path,
                         inputs=params.inputs,
                         max_len=params.sequence_length
                         )

    with open(params.results_path / 'gen_text.txt', 'w') as f:
        f.write(gen_text)
    