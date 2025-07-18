import numpy as np

class LogisticRegression:

    def __init__(self, X, y, alpha, max_iterations, print_iterations):
        self.theta = np.zeros(X.shape[1])
        self.theta = gradient_descent(self.theta, X, y, alpha, max_iterations,
                                      print_iterations)
        

    def sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def cost_function(theta, X, y):
        epsilon = 1e-10

        h = np.clip(LogisticRegression.sigmoid(np.dot(X, theta)), epsilon, 1 - epsilon)
        cost = -y * np.log(h) - (1 - y) * np.log(1 - h)

        return np.mean(cost)
    
    def gradient_update(theta, X, y):
        epsilon = 1e-10
        m = X.shape[0]
        h = np.clip(LogisticRegression.sigmoid(np.dot(X, theta)), epsilon, 1 - epsilon)
        grad = np.dot(X.T, h - y)
        return grad / m

    def gradient_descent(theta, X, y, alpha, max_iterations, print_iterations):
        iteration = 0
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        while(iteration < max_iterations):
            iteration += 1
            gradient = LogisticRegression.gradient_update(theta, X, y)
            theta -= alpha * gradient

            if iteration % print_iterations == 0 or iteration == 1:
                cost = LogisticRegression.cost_function(theta, X, y)
                print ("[ Iteration", iteration, "]", "cost =", cost)

        return theta

    def predict(X):
        theta = self.theta
        epsilon = 1e-10
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        probabilities = np.clip(LogisticRegression.sigmoid(np.dot(X, theta)), epsilon, 1 - epsilon)
        predicted_labels = np.where(probabilities >= 0.5, 1, 0)

        return probabilities, 1*predicted_labels
