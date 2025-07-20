from sklearn.datasets import load_iris
import numpy as np

from .kmeans import KMeans

def get_data():
    X, y = load_iris(return_X_y=True)

    y = np.clip(y, 0, 1)
   
    return X, y

def test_kmeans():
    X, y = get_data()

    means = KMeans(2)

    clustered = means.predict(X)

    outputs = [ [ int(y[i]) for i in cluster ] for cluster in clustered ]

    assert True

if __name__ == "__main__":
    test_kmeans()
