from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required
from webapp.static.model.textsim.search_index import index_searcher

results_bp = Blueprint('results', __name__)


@results_bp.route('/results', methods=['GET', 'POST'])
@login_required
def results(): 
    if request.method == 'POST':
        inputs = request.form
        if inputs['key_terms']:
            results = index_searcher(query_string=inputs['key_terms'])
            return render_template('results.html', inputs=inputs, results=results)
        return redirect(url_for('main.index'))