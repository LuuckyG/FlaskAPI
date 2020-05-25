from functools import wraps
from flask_mail import Message
from flask import url_for, request, redirect, session

from webapp.users.models import User

def requires_access_level(access_level):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('email'):
                return redirect(url_for('users.login'))

            user = User.find_by_email(session['email'])
            if not user.allowed(access_level):
                return redirect(url_for('users.profile', message="You do not have access to that page. Sorry!"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', 
                  sender='noreply@demo.com', 
                  recipients=[user.email])
    
    msg.body = f""" To reset your password, visit the following link:
{url_for('users.reset_token', token=token, _external=True)}

If you did not make this request, then simply ignore this email and no changes will be made.
"""
    mail.send(msg)
