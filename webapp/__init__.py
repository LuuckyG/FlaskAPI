from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_whooshee import Whooshee

db = SQLAlchemy()
whooshee = Whooshee()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b33282311698b5bb8f1979de49f5b167'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///webapp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
whooshee.init_app(app)

from src.webapp import routes