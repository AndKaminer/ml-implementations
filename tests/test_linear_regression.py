import numpy as np
import pytest
from ml_serving.flaskr.models.linear_regression import LinearRegression


def generate_synthetic_data(n_samples=100, noise_std=0.1):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    true_beta = np.array([3.0, 5.0])  # bias = 3, slope = 5
    y = true_beta[0] + true_beta[1] * X[:, 0] + np.random.normal(0, noise_std, n_samples)
    return X, y


def test_prediction_accuracy():
    X, y = generate_synthetic_data()
    model = LinearRegression(X, y, ALPHA=0.1, N_ITER=1000)

    x_test = np.array([1.5])
    y_pred = model.predict(x_test)

    assert isinstance(y_pred, float)
    assert 10.0 < y_pred < 12.0


def test_batch_predict_output_shape():
    X, y = generate_synthetic_data()
    model = LinearRegression(X, y, ALPHA=0.05, N_ITER=500)

    predictions = model.batch_predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape


def test_cost_function_decreasing():
    X, y = generate_synthetic_data()
    model = LinearRegression(X, y, ALPHA=0.1, N_ITER=1)

    X_std = model.standardize_features(X)
    X_bias = np.column_stack((np.ones(X.shape[0]), X_std))
    beta_init = np.zeros(X_bias.shape[1])

    cost_before = model.cost_function(beta_init, X_bias, y)
    beta_after = beta_init - 0.1 * model.gradient_update(beta_init, X_bias, y)
    cost_after = model.cost_function(beta_after, X_bias, y)

    assert cost_after < cost_before


def test_standardize_features_zero_mean_unit_variance():
    X = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    model = LinearRegression(X, np.array([1.0, 2.0, 3.0]), ALPHA=0.1, N_ITER=1)
    X_std = model.standardize_features(X)

    mean = np.mean(X_std, axis=0)
    std = np.std(X_std, axis=0)

    assert np.allclose(mean, 0.0, atol=1e-7)
    assert np.allclose(std, 1.0, atol=1e-7)
