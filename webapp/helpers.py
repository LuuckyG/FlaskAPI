import pandas as pd

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

from src.webapp import db

def create_table_from_excel(file_name, table_name='wbso'):
    df = pd.read_excel(file_name)
    engine = create_engine('sqlite:///webapp.db')
    df.to_sql(table_name, con=engine, index_label='id', if_exists='replace')
