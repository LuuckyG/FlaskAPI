from flask import Flask
from flask_mail import Mail
from flask_admin import Admin
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
# from flask_whooshee import Whooshee
from flask_sqlalchemy import SQLAlchemy

from webapp.config import Config

admin = Admin(template_mode='bootstrap3')
db = SQLAlchemy()
bcrypt = Bcrypt()
mail = Mail()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    admin.init_app(app)
    db.init_app(app)
    bcrypt.init_app(app)
    mail.init_app(app)
    login_manager.init_app(app)

    from webapp.main.routes import main
    from webapp.users.routes import users
    from webapp.searches.routes import searches
    from webapp.errors.handlers import errors

    app.register_blueprint(main)
    app.register_blueprint(users)
    app.register_blueprint(searches)
    app.register_blueprint(errors)

    # with app.app_context():
    #     db.drop_all()
    #     db.create_all()

    return app
