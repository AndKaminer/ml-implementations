from flask import Blueprint, request, jsonify
from ..utils import validate_input
from ..schemas import InferenceInput
from ..model_directory import ModelDirectory

def register_routes(bp):
    @bp.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        input_data = validate_input(data, InferenceInput)
        try:
            model = ModelDirectory.get_model(input_data.model_type, input_data.model_id)
            model_input = np.array(input_data.features)
            y = model.predict(model_input)
            return str(y)

        except Exception as e:
            print(e)

            return "failure"
