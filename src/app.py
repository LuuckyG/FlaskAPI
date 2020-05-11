# Serve model as a flask application
import json
import pickle
import numpy as np

from tensorflow.keras.models import load_model, model_from_json
from flask import Flask, request, render_template, url_for

from .model.textgen.textgenrnn import textgenrnn
from .forms import RegistrationForm, LoginForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b33282311698b5bb8f1979de49f5b167'

model = None

def load_model():
    global model
    model_folder = './model/results/char_l30_d2_w128_18-54'
    model = textgenrnn(model_folder=model_folder,
                       weights_path=(model_folder + '/weights.hdf5'),
                       vocab_path=(model_folder + '/vocab.json'),
                       config_path=(model_folder + '/config.json'))

    model.load(weights_path=(model_folder + '/weights.hdf5'))


@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/register')
# def register():
#     form = RegistrationForm()
#     return render_template('register.html', title='Register', form=form)

# @app.route('/login')
# def register():
#     form = LoginForm()
#     return render_template('login.html', title='Login', form=form)


@app.route('/tool')
def prediction():
    # Get input text
    input_text = request.form.get("input")

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
