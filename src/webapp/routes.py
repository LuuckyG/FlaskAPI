import json
import email_validator
import numpy as np

from tensorflow.keras.models import load_model, model_from_json
from flask import Flask, request, render_template, url_for, flash, redirect

from src.webapp import app
from src.webapp.forms import SearchForm
from src.model.textgen.textgenrnn import textgenrnn
from src.model.textsim.search_index import index_searcher


model = None
def load_model():
    global model
    model_folder = './model/results/char_l30_d2_w128_18-54'
    model = textgenrnn(model_folder=model_folder,
                       weights_path=(model_folder + '/weights.hdf5'),
                       vocab_path=(model_folder + '/vocab.json'),
                       config_path=(model_folder + '/config.json'))

    model.load(weights_path=(model_folder + '/weights.hdf5'))


@app.route('/', methods=['GET', 'POST'])
def home():
    form = SearchForm()

    if request.method == 'POST':
        search_query = request.form['key_terms']
        results = index_searcher(query_string=search_query)
        return redirect(url_for('results', results=results))

    # if form.validate_on_submit():
    #     inputs = request.ge
    #     return redirect(url_for('results'), form=form)
    return render_template('index.html', form=form)

@app.route('/results', methods=['GET', 'POST'])
def results():
    return render_template('results.html', results=results)


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