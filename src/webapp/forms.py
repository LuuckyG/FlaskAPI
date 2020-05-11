from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo


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
