import numpy as np
from typing import Optional

class MultiNaiveBayes:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.n_samples: int = X.shape[0]
        self.n_features: int = X.shape[1]
        self.classes: np.ndarray = np.unique(y)
        self.n_classes: int = self.classes.shape[0]
        self.feature_probs: np.ndarray = np.ones((self.n_classes, self.n_features), dtype=np.float64)
        self.prior_probs: np.ndarray = np.zeros(self.n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_class = X[y == cls]
            word_count = np.sum(X_class)
            self.prior_probs[idx] = X_class.shape[0] / self.n_samples
            self.feature_probs[idx] += np.sum(X_class, axis=0)
            self.feature_probs[idx] /= (word_count + self.n_features)

    def predict(self, x: np.ndarray) -> int:
        proportional_probabilities = np.zeros(self.n_classes, dtype=np.float64)

        for i in range(self.n_classes):
            weighted_feature_probs = (x * self.feature_probs[i])
            weighted_feature_probs = weighted_feature_probs[weighted_feature_probs > 0]
            p1 = np.log(self.prior_probs[i])
            p2 = np.sum(np.log(weighted_feature_probs))
            prob = p1 + p2
            proportional_probabilities[i] = prob

        return int(np.argmax(proportional_probabilities))
        
    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        outputs = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)

        return outputs

