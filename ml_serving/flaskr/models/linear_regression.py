import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ALPHA: float,
        N_ITER: int,
    ) -> None:
        self.beta: np.ndarray = self.gradient_descent(X, y, ALPHA, N_ITER)

    def predict(self, x: np.ndarray) -> float:
        x_std = (x - self.mean) / self.std
        bias = self.beta[0]
        return float(bias + np.dot(self.beta[1:], x_std))

    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        outputs = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)
        return outputs

    def cost_function(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        e = y - np.dot(X, beta)
        return float(np.mean(np.dot(e.T, e)))

    def gradient_update(self, beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return (2 / n) * np.dot(X.T, np.dot(X, beta) - y)

    def gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        n_iter: int,
    ) -> np.ndarray:
        iteration = 0
        beta = np.zeros(X.shape[1] + 1)
        num_features = X.shape[1]

        X = self.standardize_features(X)
        X = np.column_stack((np.ones(X.shape[0]), X))  # bias term

        while iteration < n_iter:
            gradient = self.gradient_update(beta, X, y)
            beta = beta - alpha * gradient
            iteration += 1

        return beta

    def standardize_features(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std
