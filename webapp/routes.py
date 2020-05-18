from flask import request, render_template, url_for, flash, redirect, session
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

        # user = User(username=form.username.data,
        #             email='test@example.com',
        #             password=form.password.data)
        # login_user(user, remember=form.remember.data)
        # return redirect(url_for("index"))

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

@app.route("/register", methods=["GET", "POST"])
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
def results(): 
    if request.method == 'POST':
        inputs = request.form
        if inputs['key_terms']:
            results = index_searcher(query_string=inputs['key_terms'])
            return render_template('results.html', inputs=inputs, results=results)
        return redirect(url_for('index'))
