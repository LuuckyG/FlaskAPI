import os

from flask import Flask
from flask_mail import Mail
from flask_admin import Admin
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from flask_login import LoginManager
# from flask_whooshee import Whooshee
from flask_sqlalchemy import SQLAlchemy

# from logging.handlers import SMTPHandler, RotatingFileHandler

from webapp.config import Config

# Initialize
mail = Mail()
db = SQLAlchemy()
bcrypt = Bcrypt()
migrate = Migrate()
admin = Admin(template_mode='bootstrap3')

login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'


def create_app(config_class=Config):
    """
    Create Flask application from config object.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    mail.init_app(app)
    admin.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    
    login_manager.init_app(app)

    from webapp.main.routes import main
    from webapp.users.routes import users
    from webapp.searches.routes import searches
    from webapp.errors.handlers import errors

    app.register_blueprint(main)
    app.register_blueprint(users)
    app.register_blueprint(searches)
    app.register_blueprint(errors)

    # # Reset database
    # TODO: Add database migrations functionality
    # with app.app_context():
    #     db.drop_all()
    #     db.create_all()

    # if not app.debug:
    #     if app.config['MAIL_SERVER']:
    #         auth = None
            
    #         if app.config['MAIL_USERNAME'] or app.config['MAIL_PASSWORD']:
    #             auth = (app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])

    #         secure = None

    #         if app.config['MAIL_USE_TLS']:
    #             secure = ()

    #         mail_handler = SMTPHandler(
    #             mailhost=(app.config['MAIL_SERVER'], app.config['MAIL_PORT']),
    #             fromaddr='no-reply@' + app.config['MAIL_SERVER'],
    #             toaddrs=app.config['ADMINS'], subject='Microblog Failure',
    #             credentials=auth, secure=secure)
    #         mail_handler.setLevel(logging.ERROR)
    #         app.logger.addHandler(mail_handler)

    #     # Log app events when in production
    #     if not os.path.exists('logs'):
    #         os.mkdir('logs')
    #     file_handler = RotatingFileHandler('logs/wbso_tool.log', maxBytes=10240,
    #                                     backupCount=10)
    #     file_handler.setFormatter(logging.Formatter(
    #         '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    #     file_handler.setLevel(logging.INFO)
    #     app.logger.addHandler(file_handler)

    #     app.logger.setLevel(logging.INFO)
    #     app.logger.info('Evolvalor WBSO Tool')

    return app
