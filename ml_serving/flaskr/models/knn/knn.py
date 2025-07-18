import numpy as np

import heapq

class KNN:
    def __init__(self, X, y, k):
        self.k = k
        self.X = X
        self.y = y
        self.clusters = np.unique(y)

    def predict(self, x):
        heap = []

        for i, point in enumerate(self.X):
            to_push = (-KNN.distance(x, point), self.y[i]) # negative because we want maxheap
            heapq.heappush(heap, to_push)
            if len(heap) > self.k:
                heapq.heappop(heap)

        counts = { cluster : 0 for cluster in self.clusters }


        for _, label in heap:
            counts[label] += 1

        out = max([ cluster for cluster in self.clusters ], key=lambda cluster : counts[cluster])
        return out
    
    def batch_predict(self, X):
        output = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            output[i] = self.predict(x)

        return output
        
    def distance(x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)
