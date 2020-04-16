# Serve model as a flask application
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from .model.textgenrnn import textgenrnn

model = None
app = Flask(__name__)


def load_model(model_folder):
    global model
    # model variable refers to the global variable

    model = textgenrnn(model_folder=model_folder,
                       weights_path=(model_folder + '/weights.hdf5')
                       vocab_path=(model_folder + 'vocab.json')
                       config_path= (model_folder + 'config.json'))

    model.load(weights_path=(model_folder + '/weights.hdf5'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/tool', methods=['POST'])
def get_prediction():
    
    if request.method == 'POST':
        
        # Get input text
        input_text = ...

        # Get model configuration
        if model.config['line_delimited']:
            n = 1000
            max_gen_length = 60 if model.config['word_level'] else 1500
        else:
            n = 1
            max_gen_length = 1000 if model.config['word_level'] else 1500

        temperature =[float(t) for t in model.config['temperature']]

        # Generate text
        gen_text = model.generate(temperature=temperature,
                                  prefix=input_text,
                                  n=n,
                                  max_gen_length=max_gen_length)
        
        return gen_text


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(debug=True)
