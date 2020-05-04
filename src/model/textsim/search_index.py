from whoosh import qparser
from whoosh.index import open_dir


def index_searcher(dirname="src/model/textsim/indexdir", 
                   query_string=None, top_n=5,
                   search_fields=['full_text']):

    ix = open_dir(dirname)
    og = qparser.OrGroup.factory(0.9)

    # Search results
    with ix.searcher() as searcher:
        
        mp = qparser.MultifieldParser(search_fields, ix.schema, group=og)
        my_query = mp.parse(query_string)

        results = searcher.search(my_query, limit=top_n)

        # Generate keywords
        keywords = [keyword for keyword, score
            in results.key_terms("full_text", docs=10, numterms=5)]

        # Print top 'N' results
        if len(results) > 0:
            for hit in results:
                print('File: {}\nTitel: {}\nScore: {}\n'.format(hit['path'], hit['title'], str(hit.score)))
                print('')

            return results


if __name__ == '__main__':
    search_query = input('Type your search here:')
    index_searcher(query_string=search_query)
