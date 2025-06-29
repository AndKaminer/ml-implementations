from flask import Flask, Blueprint, render_template

bp = Blueprint("api", __name__)

@bp.route("/ml")
def ml():
    return "hello"
