from pickle import load
from pathlib import Path
from tensorflow.keras.models import load_model

from .model import generate_text
from .hparams import HParams


def float_input(s):
    return map(float, s.split(','))


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    arg_parser.add_argument('--model_path', type=Path,
                            help="Path to pretrained model (hdf5 format).")
    arg_parser.add_argument('--tokenizer_path', type=Path,
                            help="Path where the tokenizer can be found.")
    arg_parser.add_argument('--results_path', type=Path,
                            help="Folder where the generated text should be saved to.")
    arg_parser.add_argument('--inputs', type=str,
                            help="Provide input string for the model to predict the next words.")

    # Optional parameters
    arg_parser.add_argument('--sentence_length', type=int, default=20,
                                help="The length of the sentence (in words) used for training.")
    arg_parser.add_argument('--sequence_length', type=int, default=100,
                            help="The length of the sentence generated.")

    return parser.parse_args()


def inference(model_path, tokenizer_path, results_path, inputs, sequence_len, max_len):
    """
    Function to load in pre-trained model and generate text based on inputs.
    The generated text is saved in .txt file.

    Args:
        model_path: Path to pretrained model (hdf5 format).
        tokenizer_path: Path where the tokenizer can be found.
        results_path: Folder where the generated text should be saved to.
        inputs: Provide input string for the model to predict the next words.
        sequence_len: The length of the sentence (in words) used for training.
        max_len: The number of words of the sentence generated.
    
    Returns:
        Generated text.    
    """

    model = load_model(model_path)
    with open(config['tokenizer_path'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return generate_text(model, tokenizer, inputs, reverse_word_map, max_len=max_len, **kwargs)


if __name__ == "__main__":
    params = get_parser()

    gen_text = inference(model_path=params.model_path,
                         tokenizer_path=params.tokenizer_path,
                         results_path=params.results_path,
                         inputs=params.inputs,
                         sequence_len=params.sequence_len,
                         max_len=params.sequence_length
                         )

    with open(params.results_path + '/gen_text.txt', 'w') as f:
        f.write(gen_text)
    