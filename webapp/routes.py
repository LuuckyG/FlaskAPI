from webapp import app, db, bcrypt, admin
from webapp.models import User, SearchQuery, SearchResult
from webapp.forms import LoginForm, RegistrationForm, SearchForm
from webapp.static.model.textsim.search_index import index_searcher

from flask import redirect, render_template, url_for, request, flash, session, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from flask_admin.contrib.sqla import ModelView
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(SearchQuery, db.session))
admin.add_view(ModelView(SearchResult, db.session))

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route('/', methods=['GET', 'POST'])
@login_required
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
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for("index"))
        else:
            flash('Login Unsuccesful. Please check username and password', 'danger')
    return render_template("login.html", title='Login', form=form)

@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()
    logout_user()

    # Redirect user to login form
    return redirect(url_for('index'))

@app.route("/register")
def register():
    """Register user"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        # Add user to database
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        # Redirect user to home page
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for("index"))
    return render_template('register.html', title='Register', form=form)

@app.route('/results', methods=['GET', 'POST'])
@login_required
def results(): 
    if request.method == 'POST':
        inputs = request.form
        if inputs['key_terms']:
            results = index_searcher(query_string=inputs['key_terms'])
            return render_template('results.html', inputs=inputs, results=results)
        return redirect(url_for('index'))
