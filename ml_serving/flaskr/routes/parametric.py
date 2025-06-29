from flask import Blueprint, requesst, jsonify
from ..utils import validate_input
from .schemas import InferenceInput
from ..model_directory import ModelDirectory

def register_routes(bp):
    @bp.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        input_data = validate_input(data, InferenceInput)
        return str(input_data.model_type) + " " + str(input_data.features)
