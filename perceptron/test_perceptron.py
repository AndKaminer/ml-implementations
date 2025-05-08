from sklearn.datasets import load_iris
import numpy as np

from perceptron import Perceptron

def get_data():
    X, y = load_iris(return_X_y=True)

    y = np.clip(y, 0, 1)
   
    return X, y

def main():
    X, y = get_data()

    percep = Perceptron()
    percep.fit(X, y, 1000)

    res = percep.batch_predict(X)
    print(res)
    print(y)
    print(np.sum(res == y) / y.shape[0])

if __name__ == "__main__":
    main()
