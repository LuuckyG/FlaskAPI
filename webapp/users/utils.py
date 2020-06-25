from webapp import mail
from flask_mail import Message
from flask import url_for, current_app, render_template


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', 
                  sender=current_app.config['ADMINS'][0], 
                  recipients=[user.email])
    
    msg.body = f""" To reset your password, visit the following link:
{url_for('users.reset_token', token=token, _external=True)}

If you did not make this request, then simply ignore this email and no changes will be made.
"""
    mail.send(msg)

