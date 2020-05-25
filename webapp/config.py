import os

class Config:
    """ Flask application config """
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'

    # Flask-SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-Admin settings
    FLASK_ADMIN_SWATCH = 'cerulean'

    # Flask-Mail SMTP server settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('EMAIL_USER')
    MAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    ADMINS = os.environ.get('ADMINS')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

    # # Flask-User settings
    # USER_APP_NAME = "WBSO Search Engine"
    # USER_ENABLE_EMAIL = True
    # USER_ENABLE_USERNAME = False
    # USER_EMAIL_SENDER_NAME = USER_APP_NAME
    # USER_EMAIL_SENDER_EMAIL = "noreply@example.com"
