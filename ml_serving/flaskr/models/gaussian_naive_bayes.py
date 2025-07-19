import numpy as np
from typing import Optional
from numpy.typing import NDArray

class GaussianNaiveBayes:
    n_samples: int
    n_features: int
    n_classes: int
    classes: NDArray[np.int_]
    means: NDArray[np.float64]
    vars: NDArray[np.float64]
    prior_probs: NDArray[np.float64]

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> None:
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = self.classes.shape[0]

        self.means = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.vars = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.prior_probs = np.zeros(self.n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_class = X[y == cls]
            self.means[idx, :] = X_class.mean(axis=0)
            self.vars[idx, :] = X_class.var(axis=0)
            self.prior_probs[idx] = X_class.shape[0] / self.n_samples

    def log_gauss_pdf(self, x: NDArray[np.float64], i: int) -> NDArray[np.float64]:
        mean = self.means[i]
        var = self.vars[i] + 1e-9
        return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict(self, x: NDArray[np.float64]) -> int:
        proportional_log_probs = np.zeros(self.n_classes, dtype=np.float64)

        for i in range(self.n_classes):
            log_prior = np.log(self.prior_probs[i])
            log_likelihood = np.sum(self.log_gauss_pdf(x, i))
            proportional_log_probs[i] = log_prior + log_likelihood

        return int(np.argmax(proportional_log_probs))

    def batch_predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        outputs: NDArray[np.int_] = np.zeros(X.shape[0], dtype=np.int_)
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)
        return outputs
