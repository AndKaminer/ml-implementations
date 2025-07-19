import numpy as np
from numpy.typing import NDArray

class Perceptron:

    def __init__(self, X: NDArray[np.float64], y: NDArray[np.int64], max_iters: int) -> None:
        self.n_examples: int
        self.n_features: int
        self.w: NDArray[np.float64]

        self.n_examples, self.n_features = X.shape
        self.w = np.zeros(self.n_features, dtype=np.float64)
        iterations = 0
        success = False

        while iterations < max_iters and not success:
            success = True
            iterations += 1
            for idx, x in enumerate(X):
                res = np.dot(x, self.w)
                if y[idx] == 1 and res <= 0:
                    self.w += x
                    success = False
                elif y[idx] == 0 and res > 0:
                    self.w -= x
                    success = False

    def predict(self, x: NDArray[np.float64]) -> int:
        return 1 if np.dot(x, self.w) > 0 else 0

    def batch_predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        outputs = np.zeros(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)
        return outputs

