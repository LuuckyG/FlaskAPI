from flask import request, render_template, url_for, flash, redirect
from flask_login import current_user, login_user, logout_user, login_required
from flask_admin.contrib.sqla import ModelView

from webapp import app, db, bcrypt, admin
from webapp.models import User, SearchQuery, SearchResult
from webapp.forms import SearchForm, LoginForm
from webapp.static.model.textsim.search_index import index_searcher


admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(SearchQuery, db.session))
admin.add_view(ModelView(SearchResult, db.session))


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm()
    return render_template('index.html', form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        # Query database for username
        user = User(username=form.username.data,
                    email='test@example.com',
                    password=form.password.data)
        login_user(user, remember=form.remember.data)
        return redirect(url_for("index"))

        # if user and bcrypt.check_password_hash(user.password, form.password.data):
            # login_user(user, remember=form.remember.data)
            # next_page = request.args.get('next')
            # return redirect(next_page) if next_page else redirect(url_for("index"))
        # else:
        #     flash('Login Unsuccesful. Please check username and password', 'danger')
    return render_template("login.html", title='Login', form=form)

@app.route('/results', methods=['GET', 'POST'])
def results(): 
    if request.method == 'POST':
        inputs = request.form
        if inputs['key_terms']:
            results = index_searcher(query_string=inputs['key_terms'])
            return render_template('results.html', inputs=inputs, results=results)
        return redirect(url_for('index'))
