# Serve model as a flask application
import json
import pickle
import numpy as np

from tensorflow.keras.models import load_model, model_from_json
from flask import Flask, request, render_template
from .model.textgenrnn import textgenrnn


app = Flask(__name__)

model = None

# def load_model():
#     global model
#     model_folder = './results/char_l30_d2_w128_18-54'

#     # load json and create model
#     with open(model_folder + '/model.json', 'r'):
#         model_json = file.read()
    
#     model = model_from_json(model_json)

#     # load weights
#     model.load_weights(h5_file)


def load_model():
    global model
    model_folder = './results/char_l30_d2_w128_18-54'
    model = textgenrnn(model_folder=model_folder,
                       weights_path=(model_folder + '/weights.hdf5'),
                       vocab_path=(model_folder + '/vocab.json'),
                       config_path=(model_folder + '/config.json'))

    model.load(weights_path=(model_folder + '/weights.hdf5'))
    print(model)

@app.route('/')
def home():
    load_model()
    print(model)
    return render_template('home.html')


@app.route('/tool')
def prediction():
    # Get input text
    input_text = request.form.get("input")
    # model = load_model()

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
                              max_gen_length=max_gen_length)[0]
    
    if not input_text:
        input_text = ''

    return render_template("tool.html", gen_text=gen_text, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)
