from .models import KMeans, KNN, LinearRegression, LogisticRegression
from .models import GaussianNaiveBayes, MultiNaiveBayes, Perceptron, SVM

class ModelDirectoryError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelDirectory:
    model_registry = {
        "kmeans": {1: KMeans(3)}
    } # model_type -> ({model_id -> model})

    model_type_mapping = {
        "kmeans": KMeans,
        "knn": KNN,
        "linear_regression": LinearRegression,
        "logistic_regression": LogisticRegression,
        "gauss_naive_bayes": GaussianNaiveBayes,
        "multi_naive_bayes": MultiNaiveBayes,
        "perceptron": Perceptron,
        "svm": SVM,
    }
    
    @classmethod
    def get_model(cls, model_type, model_id=0):
        if model_type not in cls.model_registry:
            raise ModelDirectoryError("Invalid model type")
    
        dict_output = cls.model_registry[model_type]
        if model_id not in dict_output:
            raise ModelDirectoryError("Invalid model id")

        return dict_output[model_id]

    @classmethod
    def register_model(cls, model_type, kwargs, model_id=0):
        if model_type not in cls.model_type_mapping:
            raise ModelDirectoryError("Model type does not exist")

        try:
            model_instance = cls.model_type_mapping[model_type](**kwargs)
        except Exception as e:
            print(e)
            raise ModelDirectoryError(
                    "Could not create instance of model with provided arguments"
            )

        if model_type not in cls.model_registry:
            cls.model_registry[model_type] = {}

        dict_output = cls.model_registry[model_type]
        if model_id in dict_output:
            raise ModelDirectoryError("Model id already exists")
        
        dict_output[model_id] = model_instance

    @classmethod
    def delete_model(cls, model_type, model_id=None):
        if model_type not in cls.model_registry:
            raise ModelDirectoryError("Invalid model type")
    
        dict_output = cls.model_registry[model_type]
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

        if model_type not in cls.model_registry:
            return []
        
        dict_output = cls.model_registry[model_type]
        return list(dict_output.keys())
