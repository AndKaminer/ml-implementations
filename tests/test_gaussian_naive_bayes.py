import numpy as np
import pytest
from ml_serving.flaskr.models.gaussian_naive_bayes import GaussianNaiveBayes

@pytest.fixture
def simple_dataset():
    X = np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [3.0, 4.0],
        [3.1, 4.1]
    ])
    y = np.array([0, 0, 1, 1])
    return X, y

def test_fit_initializes_correctly(simple_dataset):
    X, y = simple_dataset
    model = GaussianNaiveBayes()
    model.fit(X, y)

    assert model.means.shape == (2, 2)
    assert model.vars.shape == (2, 2)
    assert model.prior_probs.shape == (2,)
    np.testing.assert_allclose(model.prior_probs.sum(), 1.0)

def test_predict_single_point(simple_dataset):
    X, y = simple_dataset
    model = GaussianNaiveBayes()
    model.fit(X, y)

    test_point = np.array([1.05, 2.05])
    pred = model.predict(test_point)
    assert pred == 0

def test_batch_predict(simple_dataset):
    X, y = simple_dataset
    model = GaussianNaiveBayes()
    model.fit(X, y)

    preds = model.batch_predict(X)
    assert preds.shape == (4,)
    assert np.all(np.isin(preds, [0, 1]))

def test_model_handles_small_variances(recwarn):
    X = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [10.0, 10.0],
        [10.0, 10.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    pred = model.predict(np.array([1.0, 1.0]))
    assert pred == 0

    pred2 = model.predict(np.array([10.0, 10.0]))
    assert pred2 == 1

    # Ensure no warnings were triggered
    assert len(recwarn) == 0

def test_high_accuracy_on_separable_data():
    X = np.vstack([
        np.random.normal(loc=0, scale=1, size=(50, 2)),
        np.random.normal(loc=5, scale=1, size=(50, 2))
    ])
    y = np.array([0] * 50 + [1] * 50)

    model = GaussianNaiveBayes()
    model.fit(X, y)
    preds = model.batch_predict(X)

    accuracy = (preds == y).mean()
    assert accuracy > 0.9
