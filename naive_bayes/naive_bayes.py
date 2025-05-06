import numpy as np
import matplotlib.pyplot as plt

class NaiveBayes:

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = self.classes.shape[0]
        self.means, self.vars = np.zeros((self.n_classes, self.n_features), dtype=np.float64), np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.prior_probs = np.zeros(self.n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_class = X[y == cls]
            self.means[idx, :] = X_class.mean(axis=0)
            self.vars[idx, :] = X_class.var(axis=0)
            self.prior_probs[idx] = X_class.shape[0] / self.n_samples

        print(self.means.shape)

    def predict(self, x):
        proportional_probabilities = np.zeros(self.n_classes, dtype=np.float64)

        for i in range(self.n_classes):
            prob = np.log(self.prior_probs[i]) + np.sum(np.log(self.gauss_pdf(x, i)))
            proportional_probabilities[i] = prob

        return np.argmax(proportional_probabilities)
        

    def gauss_pdf(self, x, i):
        mean, var = self.means[i], self.vars[i]
        return np.exp(-np.square(x - mean) / 2 * var) / np.sqrt(2 * var * np.pi)
