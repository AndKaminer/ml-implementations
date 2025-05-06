import numpy as np
import kagglehub

import shutil
import os

from naive_bayes import NaiveBayes

def main():
    
    dir_path = kagglehub.dataset_download("quantbruce/real-estate-price-prediction")
    data_path = os.path.join(dir_path, "Real estate.csv")
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    X = data[:, 3:5] # distance to nearest MRT station, number of convenience stores
    y = data[:, -1] # house price

    nb = NaiveBayes()
    nb.fit(X, y)

    print(nb.predict(X[0]))

    # shutil.rmtree(dir_path)


if __name__ == "__main__":
    main()
