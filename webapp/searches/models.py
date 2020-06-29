from datetime import datetime
from webapp import db

class SearchQuery(db.Model):
    __tablename__ = 'search_query'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.TEXT, default='x')
    zwaartepunt = db.Column(db.TEXT, default='x')
    key_terms = db.Column(db.TEXT, default='x')
    aanleiding = db.Column(db.TEXT) 
    t_knel = db.Column(db.TEXT) 
    opl = db.Column(db.TEXT) 
    prog = db.Column(db.TEXT) 
    t_nieuw = db.Column(db.TEXT) 
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    search_collection = db.relationship('SearchCollection', backref='query', lazy=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Searched using: {self.title} - {self.zwaartepunt} - {self.key_terms}"


class SearchCollection(db.Model):
    __tablename__ = 'search_collection'
    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('search_query.id'), nullable=False)
    documents = db.relationship('SearchResult', backref="collection", cascade="all, delete-orphan" , lazy='dynamic')
    
    def __repr__(self):
        return f"Result collection of query: {self.query_id}"
    

class SearchResult(db.Model):
    __tablename__ = 'search_result'
    id = db.Column(db.Integer, primary_key=True)
    section = db.Column(db.String())
    rank = db.Column(db.String())
    title = db.Column(db.String())
    path = db.Column(db.String())
    bedrijf = db.Column(db.String())
    jaar = db.Column(db.Integer)
    zwaartepunt = db.Column(db.String())
    opdrachtgever = db.Column(db.String())
    full_text = db.Column(db.TEXT)
    aanleiding = db.Column(db.TEXT)
    t_knel = db.Column(db.TEXT)
    opl = db.Column(db.TEXT)
    prog = db.Column(db.TEXT)
    nieuw = db.Column(db.TEXT)
    score = db.Column(db.Float)
    date = db.Column(db.DateTime)
    query_id = db.Column(db.Integer)
    search_collection_id = db.Column(db.Integer, db.ForeignKey('search_collection.id'), nullable=False)
    
    def __repr__(self):
        return f"Bedrijf: {self.bedrijf}\nTitel: {self.title}\nScore: {self.score}"

    def __init__(self, section, rank, title, path, 
                 bedrijf, jaar, zwaartepunt, opdrachtgever, 
                 full_text, aanleiding, t_knel, opl, prog, 
                 nieuw, score, date, query_id, search_collection_id):
        self.section = section
        self.rank = rank
        self.title = title
        self.path = path
        self.bedrijf = bedrijf
        self.jaar = jaar
        self.zwaartepunt = zwaartepunt
        self.opdrachtgever = opdrachtgever
        self.full_text = full_text
        self.aanleiding = aanleiding
        self.t_knel = t_knel
        self.opl = opl
        self.prog = prog
        self.nieuw = nieuw
        self.score = score
        self.date = date
        self.query_id = query_id
        self.search_collection_id = search_collection_id

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
        return f"Bedrijf: {self.bedrijf} - Titel: {self.title}"
