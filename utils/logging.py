import json

from pickle import load, dump

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def checkpoint_log(results_directory, loss):
    model_early_stop_filename = results_directory / 'model_checkpoint.h5'
    checkpoint_metric = 'val_' + loss
    checkpoint_mode = 'min'

    # Model checkpoint.
    checkpoint = ModelCheckpoint(str(model_early_stop_filename), 
                                monitor=checkpoint_metric, 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode=checkpoint_mode, 
                                period=1)

    # Model logger.
    csv_logger = CSVLogger(str(results_directory / 'training.log'))

    # Tensorboard logger.
    tensorboard = TensorBoard(log_dir=results_directory, 
                              write_graph=True, 
                              write_grads=False, 
                              write_images=True)

    return [checkpoint, csv_logger, tensorboard]


def save_config(model, tokenizer, cfg, reverse_word_map, results_directory):

    # Serialize model to JSON
    model_json = model.to_json()
    with open(str(results_directory / 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    
    # Save tokenizer and model
    with open(results_directory / 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    model.save(str(results_directory / 'final_model.h5'))

    # Save training configuration in JSON file
    cfg['results_directory'] = str(results_directory)
    cfg['results_path'] = str(cfg['results_path'])
    cfg['path_to_file'] = str(cfg['path_to_file'])
    cfg['model_path'] = str(cfg['model_path'])
    cfg['reverse_word_map'] = reverse_word_map
    cfg['tokenizer_path'] = str(results_directory / 'tokenizer.pkl')

    with open(str(results_directory / 'config.json'), 'w') as json_file:
        json.dump(cfg, fp=json_file, indent=4)

    return cfg