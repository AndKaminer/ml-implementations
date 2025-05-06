import numpy as np
import kagglehub

import shutil
import os
import subprocess

from logistic_regression import LogisticRegression

def main():
    
    if not os.path.exists('p0_data.txt'):
        subprocess.run(["wget", "https://raw.githubusercontent.com/cocoxu/CS4650_spring2025_projects/refs/heads/main/p0_data.txt"])

    data = np.loadtxt('p0_data.txt', delimiter=',')

    X = data[:, 0:2]
    y = data[:, 2]

    initial_theta = np.random.randn(X.shape[1] + 1)
    alpha_test = .004
    max_iter = 1000000
    print_iter = 100000

    learned_theta = LogisticRegression.gradient_descent(initial_theta, X, y, alpha_test, max_iter, print_iter)
    
    res = LogisticRegression.predict(learned_theta, X)
    correct = np.sum(res == y)
    print(correct / X.shape[0])


if __name__ == "__main__":
    main()
