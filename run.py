from webapp import db, create_app
from webapp.users.models import User, Role, UserRoles, Task
from webapp.searches.models import SearchQuery, SearchCollection, SearchResult, WBSO

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Role': Role, 'UserRoles': UserRoles, 'Task': Task,
            'SearchQuery': SearchQuery, 'SearchCollection': SearchCollection, 
            'SearchResult': SearchResult, 'WBSO': WBSO}

if __name__ == '__main__':
    app.run(debug=True)
