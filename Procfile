web: flask db upgrade; gunicorn run:app
worker: rq worker -u $REDIS_URL microblog-tasks
init: python manage.py db init
migrate: python manage.py db migrate
upgrade: python manage.py db upgrade