from datetime import datetime

from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user

from webapp import db
from webapp.utils import open_doc
from webapp.searches.forms import SearchForm
from webapp.searches.models import SearchQuery, SearchResult, SearchCollection
from webapp.static.model.textsim.search_index import index_searcher

searches = Blueprint('searches', __name__)


@searches.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    form = SearchForm()
    if form.validate_on_submit():
        q = SearchQuery(
                title=form.project_titel.data,
                zwaartepunt=form.zwaartepunt.data,
                key_terms=form.key_terms.data,
                date=datetime.utcnow(),
                user_id=current_user.id)
     
        current_user.num_searches += 1
        current_user.last_searched = datetime.utcnow()

        db.session.add(q)
        db.session.commit()

        sc = SearchCollection(query_id=q.id)
        db.session.add(sc)
        db.session.commit()

        return redirect(url_for('searches.results', query_id=q.id))

    return render_template('search.html', form=form)

@searches.route('/results', methods=['GET', 'POST'])
@login_required
def results(): 
    query = SearchQuery.query.get_or_404(int(request.args.get('query_id')))
    sc = db.session.query(SearchQuery).\
        join(SearchCollection).filter(SearchCollection.query_id==query.id).first()
    results = index_searcher(query_string=query.key_terms)

    for key in results.keys():
        if results[key]:
            for i, hit in enumerate(results[key]):
                r = SearchResult(
                        section=key,
                        rank=i+1,
                        title=hit['title'],
                        path=hit['path'],
                        bedrijf=hit['bedrijf'],
                        jaar=hit['jaar'],
                        zwaartepunt=hit['zwaartepunt'],
                        opdrachtgever=hit['opdrachtgever'],
                        full_text=hit['full_text'],
                        aanleiding=hit['aanleiding'],
                        t_knel=hit['t_knel'],
                        opl=hit['opl'],
                        prog=hit['prog'],
                        nieuw=hit['nieuw'],
                        score=hit['score'],
                        date=datetime.utcnow(),
                        query_id=query.id,
                        search_collection_id=sc.id)
                # db.session.add(r)
                # db.session.commit()

    return render_template('results.html', query=query, results=results)

@searches.route("/open_document", methods=['GET', 'POST'])
def open_document():
    # name = request.args.get('name')
    # print(name.keys())
    open_doc()
    return (''), 204