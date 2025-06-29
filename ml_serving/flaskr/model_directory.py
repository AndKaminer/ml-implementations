from ..models import KMeans

class ModelDirectoryError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelDirectory:
    cls.model_type_mapping = {} # model_type -> ({model_id -> model} if parametric else model)
    
    @classmethod
    def get_model(cls, model_type, model_id=0):
        if model_type not in cls.model_type_mapping:
            raise ModelDirectoryError("Invalid model type")
    
        dict_output = cls.model_type_mapping[model_type]
        if model_id not in dict_output:
            raise ModelDirectoryError("Invalid model id")

        return dict_output[model_id]

    @classmethod
    def register_model(cls, model_type, model_instance, model_id=0):
        if model_type not in cls.model_type_mapping:
            cls.model_type_mapping[model_type] = {}

        dict_output = cls.model_type_mapping[model_type]
        if model_id in dict_output:
            raise ModelDirectoryError("Model id already exists")

        dict_output[model_id] = model_instance

    @classmethod
    def delete_model(cls, model_type, model_id=None):
        if model_type not in cls.model_type_mapping:
            raise ModelDirectoryError("Invalid model type")
    
        dict_output = cls.model_type_mapping[model_type]
        if model_id not in dict_output:
            raise ModelDirectoryError("Model id does not exist")

        del dict_output[model_id]

    @classmethod
    def list_model_types(cls):
        return list(cls.model_type_mapping.keys())

    @classmethod
    def list_models(cls, model_type):
        if model_type not in cls.model_type_mapping:
            raise ModelDirectoryError("Invalid model type")
        
        dict_output = cls.model_type_mapping[model_type]
        if type(dict_output) == type({}):
            return list(dict_output.keys())
        else:
            return [1]


def init_model_directory():
    ModelDirectory.register_model("kmeans", KMeans(3))
