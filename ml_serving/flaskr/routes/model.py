from flask import Blueprint, request, jsonify
from ..utils import validate_input
from ..schemas import InferenceInput, RegisterInput, ListInput
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

    @bp.route("/register", methods=["POST"])
    def register():
        data = request.get_json()
        data = validate_input(data, RegisterInput)
        try:
            ModelDirectory.register_model(model_type=data.model_type,
                                          kwargs=data.kwargs,
                                          model_id=data.model_id)
        except Exception as e:
            print(e)

            return "failure"

        return "success"

    @bp.route("/list", methods=["POST"])
    def list_models():
        data = request.get_json()
        data = validate_input(data, ListInput)
        return str(ModelDirectory.list_models(data.model_type))

    @bp.route("/listtypes", methods=["GET"])
    def list_model_types():
        return str(ModelDirectory.list_model_types())
