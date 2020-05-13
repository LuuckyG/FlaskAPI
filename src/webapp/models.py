from src.webapp import db


class SearchQuery(db.Model):
    __tablename__ = 'search_query'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String())
    zwaartepunt = db.Column(db.String())
    key_terms = db.Column(db.TEXT)
    search_result = db.relationship('SearchResult', backref='query', lazy=True)

    def __repr__(self):
        return f"Searched using: {self.title} - {self.zwaartepunt} - {self.key_terms}"


class SearchResult(db.Model):
    __tablename__ = 'search_result'
    id = db.Column(db.Integer, primary_key=True)
    nr = db.Column(db.String())
    title = db.Column(db.String())
    path = db.Column(db.String())
    bedrijf = db.Column(db.String())
    jaar = db.Column(db.String())
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


def create_table_from_excel(file_name, table_name='wbso'):
    import pandas as pd
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import create_engine

    df = pd.read_excel(file_name)
    engine = create_engine('sqlite:///webapp.db')
    df.to_sql(table_name, con=engine, index_label='id', if_exists='replace')
