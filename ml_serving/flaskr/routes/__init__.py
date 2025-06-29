from flask import Blueprint

from . import parametric
from . import nonparametric

bp = Blueprint("routes", __name__)

# game plan: there will be two types of models. parametric and nonparametric.
# parametric models will have the following api setup:
# - a fit call with training data and model id
#   - this registers trains a model and registers it with that id
# - a predict call with data to predict and model id
#   - this returns the prediction

# nonparametric models will have the following api setup:
# - a predict call with data
#   - this returns the prediction
