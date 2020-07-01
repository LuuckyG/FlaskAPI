from functools import wraps
from flask import url_for, request, redirect, session, abort

from webapp import db
from webapp.users.models import User


def get_or_create(model, **filter_parameters):
    instance = model.query.filter_by(**filter_parameters).first()
    if not instance:
        instance = model(**filter_parameters)
        db.session.add(instance)
        db.session.commit()

    return instance


def requires_access_level(access_level):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('email'):
                return redirect(url_for('users.login'))

            user = User.query.filter_by(email=session['email']).first()
            if not user.allowed(access_level):
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
