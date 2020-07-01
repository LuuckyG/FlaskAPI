def combine_search_form_inputs(inputs):
    search_query = ''
    for column_input in inputs:
        if column_input != '_sa_instance_state' and column_input != 'id':
            search_query += column_input + ' '
    return search_query