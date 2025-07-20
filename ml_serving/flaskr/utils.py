from pydantic import ValidationError
from flask import jsonify
from werkzeug.exceptions import BadRequest

def validate_input(data, schema_cls):
    try:
        return schema_cls(**data)
    except ValidationError as e:
        print(e)
        raise BadRequest(e.json())
