from whoosh import qparser
from whoosh import analysis
from whoosh import fields
from whoosh.index import open_dir


def index_searcher(dirname="webapp/static/model/textsim/indexdir", 
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
        results[search_field] = []

        # Search results
        with ix.searcher() as searcher:
            result = searcher.search(my_query, limit=top_n)

            if len(result) > 0:
                for hit in result:
                    hit_fields = hit.fields()
                    hit_fields['score'] = hit.score
                    hit_fields['highlights'] = hit.highlights(search_field)
                    results[search_field].append(hit_fields)

    return results


def instant_search(query_string):
    analyzer = analysis.NgramWordAnalyzer(minsize=5)
    full_text_field = fields.TEXT(analyzer=analyzer, phrase=False)
    schema = fields.Schema(full_text=full_text_field)


if __name__ == '__main__':
    search_query = input('Type your search here:')
    results = index_searcher(query_string=search_query)
