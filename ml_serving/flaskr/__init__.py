import json
import os

from flask import Flask, render_template

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    os.makedirs(app.instance_path, exist_ok=True)
    app.config.from_file("config.json", load=json.load)
    app.secret_key = "1234"

    from .routes import bp as routes_bp
    from .routes import parametric
    from .routes import nonparametric
    parametric.register_routes(routes_bp)
    nonparametric.register_routes(routes_bp)
    app.register_blueprint(routes_bp)

    return app
