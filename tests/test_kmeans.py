import numpy as np
import pytest
from ml_serving.flaskr.models.kmeans import KMeans

def test_kmeans_basic_clustering():
    # Simple 2-cluster problem
    X = np.array([
        [1, 1],
        [1.1, 1],
        [5, 5],
        [5.1, 5.1]
    ])
    
    kmeans = KMeans(k=2)
    np.random.seed(42)  # Make test deterministic
    clusters = kmeans.predict(X, n_iters=1, max_iters=100)

    # Should form 2 clusters with 2 points each
    assert len(clusters) == 2
    assert sum(len(c) for c in clusters) == 4
    assert all(isinstance(c, list) for c in clusters)

    # Check that points are reasonably clustered
    cluster_assignments = [0] * 4
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_assignments[idx] = i
    
    # Points [0, 1] should be in the same cluster, and [2, 3] in the other
    assert cluster_assignments[0] == cluster_assignments[1]
    assert cluster_assignments[2] == cluster_assignments[3]
    assert cluster_assignments[0] != cluster_assignments[2]

def test_kmeans_returns_correct_number_of_clusters():
    X = np.random.rand(100, 3)
    k = 5
    kmeans = KMeans(k)
    np.random.seed(0)
    clusters = kmeans.predict(X)

    assert len(clusters) == k
    assert sum(len(c) for c in clusters) == 100

def test_kmeans_empty_cluster_handling():
    # If all points are the same, might get empty clusters
    X = np.array([[1.0, 1.0]] * 10)
    kmeans = KMeans(k=3)
    np.random.seed(0)
    clusters = kmeans.predict(X, n_iters=1, max_iters=10)

    assert len(clusters) == 3
    assert sum(len(c) for c in clusters) == 10

def test_kmeans_converges_on_simple_data():
    X = np.array([
        [0, 0],
        [0, 0.1],
        [10, 10],
        [10.1, 10]
    ])
    kmeans = KMeans(k=2)
    np.random.seed(1)
    clusters = kmeans.predict(X, n_iters=1, max_iters=100)

    cluster_sets = [set(c) for c in clusters]
    assert all(len(c) > 0 for c in clusters)
    assert sum(len(c) for c in clusters) == 4

def test_kmeans_dist_function():
    x1 = np.array([1.0, 1.0])
    x2 = np.array([4.0, 5.0])
    dist = KMeans.dist(x1, x2)
    expected = np.sqrt((4.0 - 1.0) ** 2 + (5.0 - 1.0) ** 2)
    print(dist)
    print(expected)
    assert np.isclose(dist, expected)

def test_get_mean_handles_empty():
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    result = KMeans._get_mean([], X)
    assert np.allclose(result,np.ones(X.shape[1]))

def test_get_mean_computes_correctly():
    X = np.array([[1.0, 1.0], [3.0, 3.0]])
    cluster = [0, 1]
    expected = np.array([2.0, 2.0])
    result = KMeans._get_mean(cluster, X)
    assert np.allclose(result, expected)

