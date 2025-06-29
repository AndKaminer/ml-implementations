import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, X, y, ALPHA, N_ITER, print_iterations, feature_plots=False):
        self.beta = self.gradient_descent(X, y, ALPHA, N_ITER, print_iterations, feature_plots)

    def cost_function(self, beta, X, y):
        e = y - np.dot(X, beta)
        return np.mean(np.dot(e.T, e))

    def gradient_update(self, beta, X, y):
        n = X.shape[0]
        return (2 / n) * (np.dot(X.T, np.dot(X, beta) - y))

    def gradient_descent(self, X, y, alpha, n_iter, print_iterations, feature_plots=False):
        iteration = 0
        beta = np.zeros(X.shape[1] + 1)
        num_features = X.shape[1]

        X = self.standardize_features(X)
        X = np.column_stack((np.ones(X.shape[0]), X)) # bias term

        while iteration < n_iter:
            gradient = self.gradient_update(beta, X, y)
            beta = beta - alpha * gradient
            iteration += 1

            if iteration % print_iterations == 0 or iteration == 1:
                
                if feature_plots:

                    fig, axes = plt.subplots(1, num_features, figsize=(8 * num_features, 8))
                    for feature_num, ax in enumerate(axes if num_features > 1 else [axes]):

                        x_axis_range = np.arange(np.min(X[:, feature_num + 1]), np.max(X[:, feature_num + 1]), 0.1)
                        ax.plot(x_axis_range, beta[0] + beta[feature_num + 1] * x_axis_range, color="red")
                        ax.scatter(X[:, feature_num + 1], y)
                        ax.set_xlabel("Feature " + str(feature_num + 1))
                        ax.set_ylabel("Median House Value ($100k)")
                    
                    fig.canvas.manager.set_window_title('Feature Plots')
                    fig.suptitle("Feature Plots")
                    plt.tight_layout()
                    plt.show()

                cost = self.cost_function(beta, X, y)
                print("[ Iteration", iteration, "]", "cost =", cost)

        print("Final cost:", self.cost_function(beta, X, y))
        return beta

    def standardize_features(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std
