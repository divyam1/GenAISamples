from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')

    with app.app_context():
        from . import routes

    return app