import numpy as np
import pytest
from ml_serving.flaskr.models.logistic_regression import LogisticRegression

def generate_linearly_separable_data(n=100, d=2, seed=42):
    np.random.seed(seed)
    X_pos = np.random.randn(n // 2, d) + 2
    X_neg = np.random.randn(n // 2, d) - 2
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n // 2), np.zeros(n // 2)))
    return X, y

def test_sigmoid_properties():
    z = np.array([-1000, 0, 1000], dtype=np.float64)
    sig = LogisticRegression.sigmoid(z)
    assert np.isclose(sig[0], 0.0, atol=1e-6)
    assert np.isclose(sig[1], 0.5)
    assert np.isclose(sig[2], 1.0, atol=1e-6)

def test_cost_function_decreases():
    X, y = generate_linearly_separable_data()
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
    theta_init = np.zeros(X_bias.shape[1])
    cost1 = LogisticRegression.cost_function(theta_init, X_bias, y)
    theta_trained = LogisticRegression.gradient_descent(theta_init.copy(), X, y, alpha=0.1, max_iterations=500)
    cost2 = LogisticRegression.cost_function(theta_trained, X_bias, y)
    assert cost2 < cost1

def test_gradient_descent_learns_correctly():
    X, y = generate_linearly_separable_data()
    model = LogisticRegression(X, y, alpha=0.1, max_iterations=1000)
    probs, preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

def test_prediction_output_shapes():
    X, y = generate_linearly_separable_data(n=20)
    model = LogisticRegression(X, y, alpha=0.1, max_iterations=500)
    probs, preds = model.predict(X)
    assert probs.shape == (20,)
    assert preds.shape == (20,)

def test_small_dataset():
    X = np.array([[0.1], [0.9]])
    y = np.array([0, 1])
    model = LogisticRegression(X, y, alpha=0.5, max_iterations=300)
    probs, preds = model.predict(X)
    assert preds[0] != preds[1]

