import os
from pathlib import Path
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


### OPENING SELECTED WBSO TENDER ###
def get_teamdrive_dir():
    """Get users home directory and combine this with teamdrive directory"""
    home = str(Path.home())
    teamdrive = r'Evolvalor\Cura Innova Ventures B.V. hodn Evolvalor Team Site - Evolvalor TeamDrive'
    return os.path.join(home, teamdrive)


def find_doc(path, name):
    """Finding document in Teamdrive"""
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def open_doc():
    """Opening selected document, if it is found document structure"""

    teamdrive = get_teamdrive_dir()
    filename = 'Aanvraag WBSO 2018 1 - 6 AM Impact.pdf'

    # Check for file extension
    if len(filename.split('.')) != 2:
        extensions = ['.doc', '.docx', '.pdf']
        for extension in extensions:
            filepath = find_doc(teamdrive, (filename + extension))
            if filepath:
                break
    else:
        filepath = find_doc(teamdrive, filename)

    if filepath:
        os.startfile(filepath, 'open')
    else:
        return 'Cannot find this document..'
