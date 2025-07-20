import numpy as np
import pytest
from ml_serving.flaskr.models.knn import KNN

def test_knn_single_prediction():
    X = np.array([
        [1, 2],
        [2, 3],
        [10, 10]
    ])
    y = np.array([0, 0, 1])
    knn = KNN(X, y, k=1)

    pred = knn.predict(np.array([1.5, 2.5]))
    assert pred == 0

def test_knn_ties_broken_by_label_order():
    X = np.array([
        [0, 0],
        [1, 1],
        [10, 10],
        [11, 11]
    ])
    y = np.array([0, 0, 1, 1])
    knn = KNN(X, y, k=2)

    pred = knn.predict(np.array([5, 5]))
    assert pred in [0, 1]

def test_knn_batch_predict():
    X_train = np.array([
        [0, 0],
        [1, 1],
        [10, 10]
    ])
    y_train = np.array([0, 0, 1])
    knn = KNN(X_train, y_train, k=1)

    X_test = np.array([
        [0.5, 0.5],
        [9, 9]
    ])

    preds = knn.batch_predict(X_test)
    assert len(preds) == 2
    assert preds[0] == 0
    assert preds[1] == 1

def test_knn_all_same_point():
    X = np.array([[1, 1]] * 5)
    y = np.array([0, 0, 0, 0, 0])
    knn = KNN(X, y, k=3)
    
    pred = knn.predict(np.array([1, 1]))
    assert pred == 0

def test_knn_with_k_greater_than_dataset():
    with pytest.raises(ValueError):
        X = np.array([[1, 1], [2, 2]])
        y = np.array([0, 1])
        knn = KNN(X, y, k=5)

def test_knn_distance_function():
    x1 = np.array([0, 0])
    x2 = np.array([3, 4])
    dist = KNN.distance(x1, x2)
    assert np.isclose(dist, 5.0)

def test_knn_predict_respects_k():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [10, 10]
    ])
    y = np.array([0, 0, 0, 1])
    knn = KNN(X, y, k=3)

    pred = knn.predict(np.array([0.1, 0.1]))
    assert pred == 0
