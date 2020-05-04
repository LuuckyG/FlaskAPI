import os

from .create_index import populate_index
from .search_index import index_searcher


def text_search(dirname=None, database=None, search_query=None, top_n=5):

    if not os.path.exists(dirname):
        # Create index
        populate_index(dirname=dirname, database=database)

    # Search index
    index_searcher(dirname=dirname, query_string=search_query, top_n=top_n)


if __name__ == '__main__':
    search_query = input('Type your search here:')
    text_search(dirname='indexdir', 
                database='../data/database.xlsx', 
                search_query=search_query)
