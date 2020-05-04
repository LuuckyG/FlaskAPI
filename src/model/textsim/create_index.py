import os
import re
import pandas as pd

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC


def populate_index(dirname='/indexdir', database='C:/Users/luukg/Documents/01_Evolvalor/FlaskAPI/src/model/data/database.xlsx'):
    """
    Create schema and index.

    Schema definition: 
        - nr: Projectnummer
        - title: Projecttitel
        - path: filename
        - Bedrijf: Company filing the tender
        - Jaar: Year of tender
        - Zwaartepunt: Zwaartepunt	
        - Opdrachtgever: Company assisting with the tender
        - full_text: Full text of tender
        - aanleiding: Aanleiding
        - t_knel: Technische knelpunten	
        - opl: Oplossingsrichting	
        - Prog: Programmeertalen, ontwikkelomgevingen en tools	
        - nieuw: Waarom technisch nieuw?
    """

    schema = Schema(nr=ID (stored=True),
                    title=TEXT (stored=True),
                    path=ID (stored=True),
                    Bedrijf=TEXT (stored=True),
                    Jaar=TEXT (stored=True),
                    Zwaartepunt=TEXT (stored=True),
                    Opdrachtgever=TEXT (stored=True),
                    full_text=TEXT,
                    aanleiding=TEXT,
                    t_knel=TEXT,
                    opl=TEXT,
                    Prog=TEXT,
                    nieuw=TEXT)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    ix = create_in(dirname, schema)

    # Get content from excel file
    db = pd.read_excel(database)
    db.fillna('None', inplace=True)
    db = get_full_text(db)

    # Create writer to add content to index
    with ix.writer() as writer:
        for i, row in db.iterrows():
            add_files(row, writer)


def add_files(row, writer):
    writer.update_document(nr=str(row['Projectnummer']),
                           title=row['Projecttitel'],
                           path=row['filename'],
                           Bedrijf=row['Bedrijf'],
                           Jaar=str(row['Jaar']),
                           Zwaartepunt=row['Zwaartepunt'],
                           Opdrachtgever=row['Opdrachtgever'],
                           full_text=row['full_text'],
                           aanleiding=row['Aanleiding'],
                           t_knel=row['Technische knelpunten'],
                           opl=row['Oplossingsrichting'],
                           Prog=row['Programmeertalen, ontwikkelomgevingen en tools'],
                           nieuw=row['Waarom technisch nieuw?'])


def get_full_text(db):
    # Get parts of tender 
    parts = ["Aanleiding", 
            "Technische knelpunten", 
            "Oplossingsrichting", 
            "Programmeertalen, ontwikkelomgevingen en tools", 	
            "Waarom technisch nieuw?"]

    full_texts = []

    for i, row in db.iterrows():
        full_text = ''
        for p in parts:
            full_text += p + '\n' + row[p]
            full_text += '\n\n'
        full_texts.append(full_text)

    # Add full text to database
    db['full_text'] = full_texts
    
    return db


if __name__ == '__main__':
    populate_index()
