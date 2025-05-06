import numpy as np
import kagglehub

import shutil
import os
import subprocess

from naive_bayes import NaiveBayes

def main():
    
    if not os.path.exists('p0_data.txt'):
        subprocess.run(["wget", "https://raw.githubusercontent.com/cocoxu/CS4650_spring2025_projects/refs/heads/main/p0_data.txt"])

    data = np.loadtxt('p0_data.txt', delimiter=',')

    X = data[:, 0:2]
    y = data[:, 2]

    nb = NaiveBayes()
    nb.fit(X, y)



    res = nb.batch_predict(X)
    correct = np.sum(res == y)
    print(correct / X.shape[0])


if __name__ == "__main__":
    main()
