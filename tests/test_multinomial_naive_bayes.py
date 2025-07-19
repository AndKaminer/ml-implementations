import numpy as np
import pytest
from ml_serving.flaskr.models.multinomial_naive_bayes import MultiNaiveBayes


def test_initialization_basic():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [1, 0, 1],
        [0, 1, 0]
    ])
    y = np.array([0, 0, 1, 1])

    model = MultiNaiveBayes(X, y)

    assert model.feature_probs.shape == (len(np.unique(y)), X.shape[1])
    assert model.prior_probs.shape == (len(np.unique(y)),)

    np.testing.assert_almost_equal(np.sum(model.prior_probs), 1.0)

    for i in range(model.n_classes):
        s = np.sum(model.feature_probs[i])
        assert np.isclose(s, 1.0, atol=1e-7)


def test_predict_simple():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0]
    ])
    y = np.array([0, 1, 0, 1])

    model = MultiNaiveBayes(X, y)

    pred = model.predict(np.array([1, 0]))
    assert pred in model.classes

    pred = model.predict(np.array([0, 1]))
    assert pred in model.classes


def test_batch_predict_consistency():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0]
    ])
    y = np.array([0, 1, 0, 1])

    model = MultiNaiveBayes(X, y)

    batch_preds = model.batch_predict(X)
    assert batch_preds.shape == (X.shape[0],)

    for i, x in enumerate(X):
        assert batch_preds[i] == model.predict(x)


def test_predict_zero_vector():
    X = np.array([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0]
    ])
    y = np.array([0, 0, 1, 1])

    model = MultiNaiveBayes(X, y)

    zero_vec = np.array([0, 0])
    pred = model.predict(zero_vec)
    assert pred in model.classes


def test_invalid_input_shape():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    model = MultiNaiveBayes(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([1, 2, 3]))
