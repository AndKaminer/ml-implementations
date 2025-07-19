import numpy as np
import heapq
from typing import List

class KNN:
    def __init__(self, X: np.ndarray, y: np.ndarray, k: int):
        if not np.issubdtype(y.dtype, np.integer):
            raise TypeError("Labels must be integers")

        self.k: int = k
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.labels: np.ndarray = np.unique(y)

        if k > X.shape[0]:
            raise ValueError("Cannot have k greater than the number of samples")

    def predict(self, x: np.ndarray) -> int:
        heap: List[tuple[float, int]] = []

        for i, point in enumerate(self.X):
            to_push = (-KNN.distance(x, point), self.y[i])
            heapq.heappush(heap, to_push)
            if len(heap) > self.k:
                heapq.heappop(heap)

        counts: dict[int, int] = {int(label): 0 for label in self.labels}

        for _, label in heap:
            counts[label] += 1

        out: int = max(self.labels, key=lambda label: counts[label])
        return int(out)

    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        output = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            output[i] = self.predict(x)
        return output

    @staticmethod
    def distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

