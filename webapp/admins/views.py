from webapp import db
from webapp.users.models import User
from webapp.searches.models import SearchQuery, SearchCollection, SearchResult, WBSO

from flask import redirect, url_for, render_template, abort
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_login import current_user


class MyAdminIndexView(AdminIndexView):

    @expose('/')
    def index(self):
        if current_user.is_authenticated:
            if not current_user.is_admin():
                abort(403)
            else:
                return self.render('/admin/index.html')
        return redirect(url_for('users.login'))


class CustomAdminView(ModelView):

    column_exclude_list = ('password',)

    def is_accessible(self):
        return current_user.is_admin()

    def inaccessible_callback(self, name, **kwargs):
        if current_user.is_authenticated:
            abort(403)
        else:
            # redirect to login page if user doesn't have access
            return redirect(url_for('users.login'))


def init_admin(app, db):

    admin = Admin(app, name='WBSO TOOL', 
                  template_mode='bootstrap3', 
                  index_view=MyAdminIndexView(url='/admin'))

    admin.add_view(CustomAdminView(User, db.session))
    admin.add_view(CustomAdminView(SearchQuery, db.session))
    admin.add_view(CustomAdminView(SearchResult, db.session))
    admin.add_view(CustomAdminView(SearchCollection, db.session))
    admin.add_view(CustomAdminView(WBSO, db.session))
