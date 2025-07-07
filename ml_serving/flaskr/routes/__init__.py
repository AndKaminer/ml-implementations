from flask import Blueprint

from . import health
from . import model

bp = Blueprint("routes", __name__)
