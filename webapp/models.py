from datetime import datetime
from flask_login import UserMixin
from webapp import db, login_manager

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    searches = db.relationship('SearchQuery', backref='searched_by', lazy=True)

    def __repr__(self):
        return f"<User {self.username}>"


class SearchQuery(db.Model):
    __tablename__ = 'search_query'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.TEXT)
    zwaartepunt = db.Column(db.TEXT)
    key_terms = db.Column(db.TEXT)
    search_result = db.relationship('SearchResult', backref='query', lazy=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Searched using: {self.title} - {self.zwaartepunt} - {self.key_terms}"


class SearchResult(db.Model):
    __tablename__ = 'search_result'
    id = db.Column(db.BigInteger, primary_key=True)
    nr = db.Column(db.String())
    title = db.Column(db.String())
    path = db.Column(db.String())
    bedrijf = db.Column(db.String())
    jaar = db.Column(db.BigInteger)
    zwaartepunt = db.Column(db.String())
    opdrachtgever = db.Column(db.String())
    full_text = db.Column(db.TEXT)
    aanleiding = db.Column(db.TEXT)
    t_knel = db.Column(db.TEXT)
    opl = db.Column(db.TEXT)
    prog = db.Column(db.TEXT)
    nieuw = db.Column(db.TEXT)
    score = db.Column(db.Float)
    query_id = db.Column(db.Integer, db.ForeignKey('search_query.id'), nullable=False)
    
    def __repr__(self):
        return f"Bedrijf: {self.bedrijf}\nTitel: {self.title}\nScore: {self.score}"


# @whooshee.register_model('nr', 'title', 'bedrijf', 'jaar', 'zwaartepunt', 
#                          'opdrachtgever', 'full_text', 'aanleiding', 't_knel', 'opl', 'prog', 'nieuw')
class WBSO(db.Model):
    __tablename__ = 'wbso'
    id = db.Column(db.BigInteger, primary_key=True)
    nr = db.Column(db.TEXT)
    title = db.Column(db.TEXT)
    path = db.Column(db.TEXT)
    bedrijf = db.Column(db.TEXT)
    jaar = db.Column(db.BigInteger)
    zwaartepunt = db.Column(db.TEXT)
    opdrachtgever = db.Column(db.TEXT)
    full_text = db.Column(db.TEXT)
    aanleiding = db.Column(db.TEXT)
    t_knel = db.Column(db.TEXT)
    opl = db.Column(db.TEXT)
    prog = db.Column(db.TEXT)
    nieuw = db.Column(db.TEXT)

    def __repr__(self):
        return f"Bedrijf: {self.bedrijf}\nTitel: {self.title}\nScore: {self.score}"
