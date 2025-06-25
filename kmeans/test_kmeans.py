from sklearn.datasets import load_iris
import numpy as np

from kmeans import KMeans

def get_data():
    X, y = load_iris(return_X_y=True)

    y = np.clip(y, 0, 1)
   
    return X, y

def main():
    X, y = get_data()

    means = KMeans(2)

    clustered = means.cluster(X)

    outputs = [ [ int(y[i]) for i in cluster ] for cluster in clustered ]

    print(outputs)

if __name__ == "__main__":
    main()
