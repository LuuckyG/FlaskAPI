import argparse
from pathlib import Path


class HParams:
    """
    Class that takes in all model training parameters.
    """

    def __init__(self):
        self.args = self.get_parser()

    def tuple_input(self, s):
        x, y = map(int, s.split(','))
        return x, y

    def get_parser(self):
        arg_parser = argparse.ArgumentParser()

        # Required arguments
        arg_parser.add_argument('--results_path', type=Path,
                                # default='C:/Users/luukg/Documents/01_Evolvalor/FlaskAPI/results',
                                default='./results',
                                help="Folder where the model results are saved.")
        arg_parser.add_argument('--path_to_file', type=Path,
                                default='database.xlsx',
                                help="Path where the full dataset can be found.")
        arg_parser.add_argument('--subject', '--s', type=str, default='Aanleiding',
                                help="Choose one of the following subjects: "
                                "  - 'Aanleiding' (default), "
                                "  - 'Technische knelpunten', "
                                "  - 'Oplossingsrichting', "
                                "  - 'Programmeertalen, ontwikkelomgevingen en tools', "
                                "  - 'Waarom technisch nieuw?'."
                                )
        arg_parser.add_argument('--zwaartepunt', '--zw', type=str, default='Programmatuur',
                                help="Choose one of the following focus areas: "
                                "'Programmatuur' (default), 'Product', or 'Productieproces'.")
        arg_parser.add_argument('--text_genrnn', action='store_false',
                                help="Select to train with text_genrnn model (default) or not.")
                                
        # Hyper parameters training
        arg_parser.add_argument('--testing', action='store_true',
                                help="Select to do an testing experiment or not (default), with Shakespear text.")
        arg_parser.add_argument('--overfitting', action='store_true',
                                help="Select to do an overfitting experiment or not (default).")
        arg_parser.add_argument('--optimizer', '--o', type=str, default='adam',
                                help="Choose one of the following optimizers: 'adam' (default), 'sgd'.")
        arg_parser.add_argument('--metrics', '--m', nargs='+', type=str, default='acc',
                                help="Choose one (or more) of the following metrics: 'acc' (default), 'mae', or 'mse'.")
        arg_parser.add_argument('--loss_type', '--lt', default='ce',
                                help="Choose one of the following loss functions: 'cross entropy' (default) or 'mse'.")
        arg_parser.add_argument('--epochs', type=int, default=200,
                                help="Number of epochs to be used for training.")
        arg_parser.add_argument('--batch_size', type=int, default=128,
                                help="Number of epochs to be used for training.")
        arg_parser.add_argument('--learning_rate', '--lr', type=float, default=0.003,
                                help="The learning rate of the optimizer.")
        arg_parser.add_argument('--seed', type=int, default=101,
                                help="The random seed used to initialize the RandomState.")
        arg_parser.add_argument('--test_size', type=float, default=0.2,
                                help="Fraction (0 < test_size < 1) of data used for testing/validation. Default: 0.2.")
        arg_parser.add_argument('--max_words', type=int, default=50000,
                                help="Max size of the dictionary.")
        arg_parser.add_argument('--line_delimited', action='store_true',
                                help="Set to True if each text has its own line in the source file (default is false).")
        arg_parser.add_argument('--gen_epochs', type=int, default=101,
                                help="Generates sample text from model after given number of epochs.")
        arg_parser.add_argument('--validation', action='store_false',
                                help="If train__size < 1.0, test on holdout dataset; will make overall training slower (default is true).")
        arg_parser.add_argument('--is_csv', action='store_true',
                                help="Set to True if file is a CSV exported from Excel/BigQuery/pandas (default is false).")
        arg_parser.add_argument('--temperature', '--t', nargs='+', type=float, default=(1.0, 0.5, 0.2),
                                help="Samples predicted probabilities of the next character to allow for the network to show 'creativity..'. \
                                    Default: (1.0, 0.5, 0.2).")

        # LSTM-Net architecture parameters
        arg_parser.add_argument('--model_name', type=str, default='textgen_model'
                                help="Name of model. Default is 'textgen_model'.")
        arg_parser.add_argument('--rnn_layers', '--rnn', type=int, default=3,
                                help="Number of rnn-based layers.")
        arg_parser.add_argument('--rnn_size', type=int, default=256,
                                help="Number of cells in each LSTM layer.")                        
        arg_parser.add_argument('--activation', '--act', type=str, default='relu',
                                help="Choose activation function for hidden layers: 'relu' (default), 'sigmoid', or 'leakyrelu'.")
        arg_parser.add_argument('--dropout', type=float, default=0.3,
                                help="Ignore a random proportion of source tokens each epoch, allowing model to generalize better.")
        arg_parser.add_argument('--word_level', action='store_true',
                                help="Set to True if want to train a word-level model (requires more data and smaller max_length) (default is false).")
        arg_parser.add_argument('--rnn_bidirectional', action='store_true',
                                help="Consider text both forwards and backward, can give a training boost (default is false).")
        arg_parser.add_argument('--epochs', type=int, default=200,
                                help="Number of epochs to be used for training.")
        arg_parser.add_argument('--epochs', type=int, default=200,
                                help="Number of epochs to be used for training.")            

        # Text generation parameters
        arg_parser.add_argument('--generate', action='store_true',
                                help="Select to also generate text after training or not (default).")
        arg_parser.add_argument('--sentence_length', type=int, default=20,
                                help="The length of the sentence (in tokens) used for training. (20-40 for characters, 5-10 for words recommended)")
        arg_parser.add_argument('--sequence_length', type=int, default=100,
                                help="The length of the sentence generated.")
        arg_parser.add_argument('--input_string', '--inp', type=str, 
                                default='De aanvrager wil een nieuw platform ontwikkelen dat ervoor zorgt dat er meer geld verdiend wordt. Dit helpt het bedrijf met de groei en continuiteit, maar zal ook een maatschappelijk voordeel hebben, namelijk ',
                                help="Provide input string for the model to predict the next words.")
        arg_parser.add_argument('--pre_trained', action='store_true',
                                help="Select to generate text with pre-trained model or not (default).")
        arg_parser.add_argument('--model_path', type=Path,
                                default=None, help="Path to pretrained model (hdf5 format).")

        return arg_parser.parse_args()
