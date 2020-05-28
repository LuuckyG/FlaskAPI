import rq
import redis

from datetime import datetime
from flask import current_app
from flask_login import UserMixin
from sqlalchemy import (Boolean, Integer, String, 
    Column, ForeignKey, DateTime)
from sqlalchemy.orm import backref, relationship
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

from webapp import db, login_manager
from webapp.searches.models import SearchQuery


ACCESS = {
    'guest': 0,
    'user': 1,
    'moderator': 2,
    'admin': 3
}


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(20), unique=True, nullable=False)
    password = Column(String(60), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_on = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_online = Column(DateTime, nullable=False, default=datetime.utcnow)
    num_searches = Column(Integer, nullable=False, default=0)
    last_searched = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(Boolean(), nullable=False, server_default='1')
    access = Column(Integer(), nullable=False, default=ACCESS['user'])
    tasks = relationship('Task', backref='user', lazy='dynamic')
    searches = relationship('SearchQuery', backref='searched_by', lazy=True)

    def __repr__(self):
        return f"{self.username}"
    
    def is_admin(self):
        return self.access == ACCESS['admin']
    
    def allowed(self, access_level):
        return self.access >= access_level
    
    def get_reset_token(self, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')
    
    @staticmethod
    def verify_reset_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)


class Task(db.Model):
    __tablename__ = 'task'
    id = Column(String(36), primary_key=True)
    name = Column(String(128), index=True)
    description = Column(String(128))
    user_id = Column(Integer, ForeignKey('user.id'))
    complete = Column(Boolean, default=False)

    def get_rq_job(self):
        try:
            rq_job = rq.job.Job.fetch(self.id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def get_progress(self):
        job = self.get_rq_job()
        return job.meta.get('progress', 0) if job is not None else 100
