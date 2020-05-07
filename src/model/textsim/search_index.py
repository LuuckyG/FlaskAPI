from whoosh import qparser
from whoosh import analysis
from whoosh import fields
from whoosh.index import open_dir


def index_searcher(dirname="src/model/textsim/indexdir", 
                   query_string=None, 
                   top_n=5,
                   search_fields=['full_text', 
                                  'aanleiding', 
                                  't_knel', 
                                  'opl', 
                                  'prog', 
                                  'nieuw']):
    """
    Function that searches all specified search fields,
    and mathces the 5 best documents in the corpus with 
    the input query string.

    Returns:
        A dictionary with the separate results.
        The keys of the dict are the names of the search fields.
    """
    ix = open_dir(dirname)
    
    results = dict()
    og = qparser.OrGroup.factory(0.9)

    for search_field in search_fields:
        # Set up parser
        parser = qparser.MultifieldParser([search_field], ix.schema, group=og)
        parser.add_plugin(qparser.FuzzyTermPlugin())
        my_query = parser.parse(query_string)

        # Search results
        with ix.searcher() as searcher:
            result = searcher.search(my_query, limit=top_n)
            results[search_field] = result

            # # Print top 'N' results
            # if len(result) > 0:
            #     for hit in result:
            #         print('Bedrijf: {}\nFile: {}\nTitel: {}\nScore: {}\n'.format(hit['bedrijf'], hit['path'], hit['title'], str(hit.score)))

            #         print(hit.highlights(search_field))
            #         print('\n')

    return results


def instant_search(query_string):
    # For example, to search the "full_text" field as the user types
    analyzer = analysis.NgramWordAnalyzer(minsize=5)
    full_text_field = fields.TEXT(analyzer=analyzer, phrase=False)
    schema = fields.Schema(full_text=full_text_field)


if __name__ == '__main__':
    search_query = input('Type your search here:')
    results = index_searcher(query_string=search_query)
