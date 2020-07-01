from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField


class UploadForm(FlaskForm):
    input_file = FileField('Upload huidige versie WBSO aanvraag', validators=[FileAllowed(['doc', 'docx'])])
    submit = SubmitField('Upload')


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
