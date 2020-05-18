from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Email, Length


class LoginForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class SearchForm(FlaskForm):
    project_titel = StringField('Projecttitel')
    zwaartepunt = StringField('Zwaartepunt')
    key_terms = StringField('Key terms')
    aanleiding = StringField('Aanleiding')
    t_knel = StringField('Technische knelpunten')
    opl = StringField('Oplossingsrichting')
    prog = StringField('Programmeertalen, ontwikkelomgevingen en tools')
    nieuw = StringField('Waarom technisch nieuw')
    submit = SubmitField('Zoek aanvragen!')
