from datetime import datetime
from flask_login import UserMixin
from flask_security import RoleMixin
from webapp import db, login_manager
from webapp.searches.models import SearchQuery

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_online = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    num_searches = db.Column(db.Integer, nullable=False, default=0)
    last_searched = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.Boolean, nullable=False, server_default='1')
    roles = db.relationship('Role', secondary='user_roles')
    searches = db.relationship('SearchQuery', backref='searched_by', lazy=True)

    def __repr__(self):
        return f"{self.username}"
    
    def is_admin(self):
        return self.access == ACCESS['admin']
    
    def allowed(self, access_level):
        return self.access >= access_level


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(50), unique=True)


class UserRoles(db.Model):
    __tablename__ = 'user_roles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id', ondelete='CASCADE'))