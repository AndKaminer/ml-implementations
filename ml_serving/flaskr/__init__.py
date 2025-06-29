from flask import Flask, render_template

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_file("config.json", load=json.load)
    app.secret_key = "1234"

    os.makedirs(app.instance_path, exist_ok=True)

    from . import api
    app.register_blueprint(api.bp)

    return app
