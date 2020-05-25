from webapp import db, admin, bcrypt, mail
from webapp.users.utils import send_reset_email
from webapp.users.models import User
from webapp.users.forms import LoginForm, RegistrationForm, RequestResetForm, ResetPasswordForm
from webapp.searches.models import SearchQuery, SearchResult, SearchCollection, WBSO

from datetime import datetime
from flask import (Blueprint, current_app, redirect, 
                    render_template, url_for, request, flash, session)
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

from flask_admin.contrib.sqla import ModelView

# Create admin views
admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(SearchQuery, db.session))
admin.add_view(ModelView(SearchResult, db.session))
admin.add_view(ModelView(SearchCollection, db.session))
admin.add_view(ModelView(WBSO, db.session))


users = Blueprint('users', __name__)


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

            # Update online status
            user.status = True
            db.session.commit()

            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for("main.index"))
        else:
            flash('Login Unsuccesful. Please check username and password', 'danger')
    return render_template("login.html", title='Login', form=form)


@users.route("/logout")
def logout():
    """Log user out"""

    # Update last seen online and change status to offline
    current_user.last_online = datetime.utcnow()
    current_user.status = False
    db.session.commit()

    # Forget any user_id
    session.clear()
    logout_user()

    # Redirect user to login form
    return redirect(url_for('users.login'))


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

        # Redirect user to login page
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for("users.login"))
    return render_template('register.html', title='Register', form=form)


@users.route('/history/<int:user_id>', methods=['GET', 'POST'])
@login_required
def history(user_id): 
    if current_user.is_authenticated:
        return render_template('history.html', user_id=current_user.id)
    return redirect(url_for('users.login'))


@users.route('/account', methods=['GET', 'POST'])
@login_required
def account(): 
    if current_user.is_authenticated:
        return render_template('account.html')
    return redirect(url_for('users.login'))


@users.route("/reset_password", methods=["GET", "POST"])
def reset_request():
    """Request password reset"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An reset password email has been send.', 'info')
        return redirect(url_for('users.login'))

    return render_template('reset_request.html', form=form) 


@users.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_token(token):
    """Reset password"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token.', 'warning')
        return redirect(url_for('users.reset_request'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        # Add user to database
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()

        # Redirect user to login page
        flash(f'Your password has been updated! You are now able to log in.', 'success')
        return redirect(url_for("users.login"))
    return render_template('reset_token.html', form=form)


@users.route('/admin', methods=['GET', 'POST'])
@login_required
def admin(): 
    if current_user.is_authenticated:
        return render_template('admin/index.html')
    return redirect(url_for('users.login'))