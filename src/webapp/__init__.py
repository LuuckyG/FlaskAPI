from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'b33282311698b5bb8f1979de49f5b167'

from src.webapp import routes