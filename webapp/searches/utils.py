FIELDS_OF_INTEREST = ['key_terms', 
                      'title', 
                      'aanleiding',
                      'opl',
                      't_knel',
                      'prog',
                      't_nieuw']


def combine_search_form_inputs(inputs):
    search_query = ''

    for field in FIELDS_OF_INTEREST:
        try:
            value = inputs[field]
            if not value == '':

                if not isinstance(value, str):
                    value = str(value)

                search_query += value + '.\n'

        except KeyError:
            continue

    return search_query