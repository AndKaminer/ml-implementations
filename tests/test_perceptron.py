import numpy as np
import pytest
from ml_serving.flaskr.models.perceptron import Perceptron

@pytest.fixture
def sample_data():
    # Linearly separable data
    X = np.array([
        [2, 3],
        [1, 1],
        [-1, -1],
        [-2, -3]
    ], dtype=np.float64)
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    return X, y

def test_fit_learns_correctly(sample_data):
    X, y = sample_data
    clf = Perceptron(X, y, max_iters=10)

    # Check if the model classifies all training points correctly
    predictions = clf.batch_predict(X)
    assert np.array_equal(predictions, y), "Model should classify all training data correctly"

def test_predict_individual(sample_data):
    X, y = sample_data
    clf = Perceptron(X, y, max_iters=10)

    for x, true_label in zip(X, y):
        pred = clf.predict(x)
        assert pred == true_label, f"Expected {true_label}, got {pred}"

def test_batch_predict_shape(sample_data):
    X, y = sample_data
    clf = Perceptron(X, y, max_iters=10)

    preds = clf.batch_predict(X)
    assert preds.shape == y.shape, "Prediction shape should match label shape"
    assert preds.dtype == np.int64, "Prediction dtype should be int64"

