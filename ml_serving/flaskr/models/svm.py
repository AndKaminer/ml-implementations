import numpy as np
from typing import Optional

class SVM:
    def __init__(self, X: np.ndarray, y: np.ndarray, num_iters: int, lr: float, lam: float) -> None:
        self.w: np.ndarray = np.random.normal(size=X.shape[-1])
        self.b: float = 0.0
        self.lr: float = lr
        self.lam: float = lam

        for _ in range(num_iters):
            for i, x_i in enumerate(X):
                if y[i] * np.dot(self.w, x_i) >= 1:
                    self.w = self.w - (self.lr * 2 * self.lam * self.w)
                else:
                    self.w = self.w - (self.lr * 2 * self.w - y[i] * x_i)
                    self.b = self.b - (self.lr * y[i])

    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        output = np.zeros(X.shape[0])

        for i, x_i in enumerate(X):
            output[i] = self.predict(x_i)

        return output

    def predict(self, x: np.ndarray) -> int:
        if np.dot(self.w, x) - self.b > 0:
            return 1
        else:
            return -1
