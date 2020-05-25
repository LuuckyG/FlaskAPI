import os

class Config:
    """ Flask application config """
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY')

    # Flask-SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-Admin settings
    FLASK_ADMIN_SWATCH = 'cerulean'

    # Flask-Mail SMTP server settings
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_SSL = True
    MAIL_USE_TLS = False
    MAIL_USERNAME = os.environ.get('EMAIL_USER')
    MAIL_PASSWORD = os.environ.get('EMAIL_PASS')
    MAIL_DEFAULT_SENDER = '"MyApp" <noreply@example.com>'

    # # Flask-User settings
    # USER_APP_NAME = "WBSO Search Engine"
    # USER_ENABLE_EMAIL = True
    # USER_ENABLE_USERNAME = False
    # USER_EMAIL_SENDER_NAME = USER_APP_NAME
    # USER_EMAIL_SENDER_EMAIL = "noreply@example.com"
