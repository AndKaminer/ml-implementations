import numpy as np

class Perceptron:

    def fit(self, X, y, max_iters):
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

    def predict(self, x):
        if np.dot(x, self.w) > 0:
            return 1
        else:
            return 0

    def batch_predict(self, X):
        outputs = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            outputs[i] = self.predict(x)

        return outputs
