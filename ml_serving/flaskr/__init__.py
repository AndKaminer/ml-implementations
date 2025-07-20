import json
import os

from flask import Flask, render_template

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    os.makedirs(app.instance_path, exist_ok=True)
    app.config.from_file(os.path.join(app.instance_path, "config.json"), load=json.load)
    app.secret_key = "1234"

    from .routes import bp as routes_bp
    from .routes import model
    from .routes import health
    model.register_routes(routes_bp)
    health.register_routes(routes_bp)
    app.register_blueprint(routes_bp)

    return app
