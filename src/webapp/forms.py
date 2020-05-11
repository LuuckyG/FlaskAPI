from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

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
