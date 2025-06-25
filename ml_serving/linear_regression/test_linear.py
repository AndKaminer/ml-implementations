import numpy as np
import kagglehub

import shutil
import os

from linear import LinearRegression


ALPHA = 0.001
N_ITER = 100000
PRINT_ITER = N_ITER / 10


def main():
    
    dir_path = kagglehub.dataset_download("quantbruce/real-estate-price-prediction")
    data_path = os.path.join(dir_path, "Real estate.csv")
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    X = data[:, 3:5] # distance to nearest MRT station, number of convenience stores
    y = data[:, -1] # house price

    lr = LinearRegression()
    lr.gradient_descent(X, y, ALPHA, N_ITER, PRINT_ITER, True)

    # shutil.rmtree(dir_path)


if __name__ == "__main__":
    main()
