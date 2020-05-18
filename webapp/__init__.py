from flask import Flask
from flask_admin import Admin
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_whooshee import Whooshee
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['SECRET_KEY'] = 'b33282311698b5bb8f1979de49f5b167'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///webapp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'

admin = Admin(app, template_mode='bootstrap3')
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

from webapp import routes