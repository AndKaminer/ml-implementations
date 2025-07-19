import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class LogisticRegression:
    def __init__(
        self, 
        X: NDArray[np.float64], 
        y: NDArray[np.float64], 
        alpha: float, 
        max_iterations: int
    ) -> None:
        self.theta: NDArray[np.float64] = np.zeros(X.shape[1] + 1)
        self.theta = LogisticRegression.gradient_descent(self.theta, X, y, alpha, max_iterations)

    @staticmethod
    def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cost_function(
        theta: NDArray[np.float64], 
        X: NDArray[np.float64], 
        y: NDArray[np.float64]
    ) -> float:
        epsilon = 1e-10
        h = np.clip(LogisticRegression.sigmoid(np.dot(X, theta)), epsilon, 1 - epsilon)
        cost = -y * np.log(h) - (1 - y) * np.log(1 - h)
        return float(np.mean(cost))

    @staticmethod
    def gradient_update(
        theta: NDArray[np.float64], 
        X: NDArray[np.float64], 
        y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        epsilon = 1e-10
        m = X.shape[0]
        h = np.clip(LogisticRegression.sigmoid(np.dot(X, theta)), epsilon, 1 - epsilon)
        grad = np.dot(X.T, h - y)
        return grad / m

    @staticmethod
    def gradient_descent(
        theta: NDArray[np.float64], 
        X: NDArray[np.float64], 
        y: NDArray[np.float64], 
        alpha: float, 
        max_iterations: int
    ) -> NDArray[np.float64]:
        iteration = 0
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        while iteration < max_iterations:
            iteration += 1
            gradient = LogisticRegression.gradient_update(theta, X, y)
            theta -= alpha * gradient
        return theta

    def predict(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        epsilon = 1e-10
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        probabilities = np.clip(LogisticRegression.sigmoid(np.dot(X, self.theta)), epsilon, 1 - epsilon)
        predicted_labels = np.where(probabilities >= 0.5, 1, 0)
        return probabilities, predicted_labels.astype(np.int64)
