import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

from .model_base_classes import Model

class KMeans(Model):
    def __init__(self, k: int) -> None:
        self.k = k

    def predict(
        self, 
        X: NDArray[np.float64], 
        n_iters: int = 10, 
        max_iters: int = 10
    ) -> List[List[int]]:
        maxes_per_dim: NDArray[np.float64] = np.max(X, axis=0)
        mins_per_dim: NDArray[np.float64] = np.min(X, axis=0)

        best_clustering: List[List[int]] = []
        best_variation: float = float("inf")
        for _ in range(n_iters):
            clustering, variation = self._cluster_iter(X, maxes_per_dim, mins_per_dim, max_iters)
            if variation < best_variation:
                best_clustering = clustering
                best_variation = variation

        return best_clustering

    @staticmethod
    def _get_mean(
        cluster: List[int], 
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        out = sum([X[c] for c in cluster])
        return out / len(cluster) if len(cluster) > 0 else np.ones(X.shape[1], dtype=np.float64)

    def _cluster_iter(
        self, 
        X: NDArray[np.float64], 
        maxes_per_dim: NDArray[np.float64], 
        mins_per_dim: NDArray[np.float64], 
        max_iters: int = 10
    ) -> Tuple[List[List[int]], float]:
        ndims: int = X.shape[1]
        iters: int = 0

        means: NDArray[np.float64] = (
            np.random.randn(self.k, ndims) * (maxes_per_dim - mins_per_dim) + mins_per_dim
        )

        last_clusters: List[List[int]] = [[] for _ in range(self.k)]

        while iters < max_iters:
            clusters: List[List[int]] = [[] for _ in range(self.k)]

            for entry_idx, x in enumerate(X):
                best_cluster: int = 0
                best_distance: float = float("inf")
                for i, m in enumerate(means):
                    dist = KMeans.dist(x, m)
                    if dist < best_distance:
                        best_cluster = i
                        best_distance = dist

                clusters[best_cluster].append(entry_idx)

            for imean in range(means.shape[0]):
                means[imean] = KMeans._get_mean(clusters[imean], X)

            if clusters == last_clusters:
                break

            last_clusters = clusters
            iters += 1

        tot: float = 0.0
        for cidx, cluster in enumerate(clusters):
            for x in cluster:
                tot += KMeans.dist(means[cidx], X[x])

        return last_clusters, tot

    @staticmethod
    def dist(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))
