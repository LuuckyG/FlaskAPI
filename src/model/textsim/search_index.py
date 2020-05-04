from whoosh.index import open_dir
from whoosh.qparser import QueryParser


def index_searcher(dirname="indexdir", query_string=None, top_n=5):

    ix = open_dir(dirname)

    # Search results
    with ix.searcher() as searcher:
        parser = QueryParser("full_text", ix.schema)
        my_query = parser.parse(query_string)
        results = searcher.search(my_query, limit=top_n)
    
    # Print N found results
    for i in range(top_n):
        print(results[i]['title'], str(results[i].score), results[i]['full_text'])


if __name__ == '__main__':
    search_query = input('Type your search here:')
    index_searcher(query_string=search_query)
