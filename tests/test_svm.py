import numpy as np
import pytest
from ml_serving.flaskr.models.svm import SVM

@pytest.fixture
def linearly_separable_data():
    # Simple 2D, linearly separable dataset
    X = np.array([
        [2, 3],
        [1, 1],
        [-1, -1],
        [-2, -3]
    ])
    y = np.array([1, 1, -1, -1])
    return X, y

def test_svm_trains_and_predicts_correctly(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVM(X, y, num_iters=100, lr=0.01, lam=0.01)

    for x_i, y_i in zip(X, y):
        pred = model.predict(x_i)
        assert pred == y_i, f"Expected {y_i}, got {pred}"

def test_batch_prediction_matches_individual(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVM(X, y, num_iters=100, lr=0.01, lam=0.01)

    batch_preds = model.batch_predict(X)
    for i in range(len(X)):
        individual_pred = model.predict(X[i])
        assert batch_preds[i] == individual_pred

def test_svm_with_zero_input():
    X = np.zeros((4, 2))
    y = np.array([1, -1, 1, -1])
    model = SVM(X, y, num_iters=10, lr=0.01, lam=0.01)

    pred = model.predict(np.zeros(2))
    assert pred == -1

def test_weights_change_during_training(linearly_separable_data):
    X, y = linearly_separable_data
    model = SVM(X, y, num_iters=1, lr=0.01, lam=0.01)

    initial_weights = np.random.normal(size=X.shape[-1])
    trained_model = SVM(X, y, num_iters=10, lr=0.01, lam=0.01)

    assert not np.allclose(initial_weights, trained_model.w)
