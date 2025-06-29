import numpy as np

class SVM:

    def __init__(self, lr, lam):
        self.w = None
        self.b = None
        self.lr = lr
        self.lam = lam

    def fit(self, X, y, num_iters=1000):
        self.w = np.random.normal(size=X.shape[-1])
        self.b = 0

        for _ in range(num_iters):
            for i, x_i in enumerate(X):
                if y[i] * np.dot(self.w, x_i) >= 1:
                    self.w = self.w - (self.lr * 2 * self.lam * self.w)
                else:
                    self.w = self.w - (self.lr * 2 * self.w - y[i] * x_i)
                    self.b = self.b - (self.lr * y[i])

    def batch_predict(self, X):
        output = np.zeros(X.shape[0])

        for i, x_i in enumerate(X):
            output[i] = self.predict(x_i)

        return output

    def predict(self, x):
        if np.dot(self.w, x) - self.b > 0:
            return 1
        else:
            return -1
