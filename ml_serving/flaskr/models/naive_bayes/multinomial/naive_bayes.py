import numpy as np
import matplotlib.pyplot as plt

class MultiNaiveBayes:

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = self.classes.shape[0]
        self.feature_probs = np.ones((self.n_classes, self.n_features), dtype=np.float64)
        self.prior_probs = np.zeros(self.n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_class = X[y == cls]
            word_count = np.sum(X_class)
            self.prior_probs[idx] = X_class.shape[0] / self.n_samples
            self.feature_probs[idx] += np.sum(X_class, axis=0)
            self.feature_probs[idx] /= (word_count + self.n_features)

    def predict(self, x):
        proportional_probabilities = np.zeros(self.n_classes, dtype=np.float64)

        for i in range(self.n_classes):
            weighted_feature_probs = (x * self.feature_probs[i])
            weighted_feature_probs = weighted_feature_probs[weighted_feature_probs > 0]
            p1 = np.log(self.prior_probs[i])
            p2 = np.sum(np.log(weighted_feature_probs))
            prob = p1 + p2
            proportional_probabilities[i] = prob

        return np.argmax(proportional_probabilities)
        
    def batch_predict(self, X):
        outputs = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)

        return outputs
