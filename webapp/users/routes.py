from webapp import db, admin, bcrypt
from webapp.users.models import User
from webapp.users.forms import LoginForm, RegistrationForm
from webapp.searches.models import SearchQuery, SearchResult, WBSO

from flask import Blueprint, current_app, redirect, render_template, url_for, request, flash, session
from flask_user import roles_required, UserManager
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

from flask_admin.contrib.sqla import ModelView

# Create admin views
admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(SearchQuery, db.session))
admin.add_view(ModelView(SearchResult, db.session))
admin.add_view(ModelView(WBSO, db.session))


users = Blueprint('users', __name__)

# Setup Flask-User and specify the User data-model
# user_manager = UserManager(current_app, db, User)

@users.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        # Query database for username
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            session['email'] = user.email
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for("main.index"))
        else:
            flash('Login Unsuccesful. Please check username and password', 'danger')
    return render_template("login.html", title='Login', form=form)

@users.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()
    logout_user()

    # Redirect user to login form
    return redirect(url_for('main.index'))

@users.route("/register", methods=['GET', 'POST'])
def register():
    """Register user"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        # Add user to database
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        # Redirect user to home page
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for("main.index"))
    return render_template('register.html', title='Register', form=form)

@users.route('/history/<int:user_id>', methods=['GET', 'POST'])
@login_required
def history(user_id): 
    if current_user.is_authenticated:
        return render_template('history.html', user_id=current_user.id)
    return redirect(url_for('main.index'))

@users.route('/account', methods=['GET', 'POST'])
@login_required
def account(): 
    if current_user.is_authenticated:
        return render_template('account.html')
    return redirect(url_for('main.index'))
