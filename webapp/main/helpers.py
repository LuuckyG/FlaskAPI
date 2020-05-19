import pandas as pd

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

from webapp import db

def create_table_from_excel(file_name, table_name='wbso'):
    df = pd.read_excel(file_name)
    engine = create_engine('sqlite:///webapp.db')
    df.to_sql(table_name, con=engine, index_label='id', if_exists='replace')

# from functools import wraps
# from flask import url_for, request, redirect, session
# from user import User

# def requires_access_level(access_level):
#     def decorator(f):
#         @wraps(f)
#         def decorated_function(*args, **kwargs):
#             if not session.get('email'):
#                 return redirect(url_for('users.login'))

#             user = User.find_by_email(session['email'])
#             elif not user.allowed(access_level):
#                 return redirect(url_for('users.profile', message="You do not have access to that page. Sorry!"))
#             return f(*args, **kwargs)
#         return decorated_function
#     return decorator

# @app.route('/control-panel')
# @requires_access_level(ACCESS['admin'])

# ACCESS = {
#     'guest': 0,
#     'user': 1,
#     'admin': 2
# }
