import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, X, n_iters=10, max_iters=10):
        maxes_per_dim = np.max(X, axis=0)
        mins_per_dim = np.min(X, axis=0)

        best_clustering = []
        best_variation = float("inf")
        for _ in range(n_iters):
           clustering, variation = self._cluster_iter(X, maxes_per_dim, mins_per_dim, max_iters)
           if variation < best_variation:
               best_clustering = clustering
               best_variation = variation

        return best_clustering

    def _get_mean(cluster, X):
        out = sum([ X[c] for c in cluster ])
        return out / len(cluster) if len(cluster) > 0 else 1
    
    def _cluster_iter(self, X, maxes_per_dim, mins_per_dim, max_iters=10):
        ndims = X.shape[1]
        iters = 0

        # randomly generate k means
        means = np.random.randn(self.k, ndims) * (maxes_per_dim - mins_per_dim) + mins_per_dim

        # for each point, find closest mean
        # then, cluster and recalc mean
        # repeat until no change

        last_clusters = [ [] for _ in range(self.k) ]
        
        while iters < max_iters:
            clusters = [ [] for _ in range(self.k) ]
            
            # cluster to means
            for entry_idx, x in enumerate(X):
                best_cluster = None
                best_distance = float("inf")
                for i, m in enumerate(means):
                    dist = KMeans.dist(x, m)
                    if dist < best_distance:
                        best_cluster = i
                        best_distance = dist

                clusters[best_cluster].append(entry_idx)

            # calc new means
            for imean in range(means.shape[0]):
                means[imean] = KMeans._get_mean(clusters[imean], X)


            # determine break
            if clusters == last_clusters:
                break

            last_clusters = clusters
            iters += 1


        # calculate variation
        tot = 0
        for cidx, cluster in enumerate(clusters):
            for x in cluster:
                tot += KMeans.dist(means[cidx], X[x])


        return last_clusters, tot

    def dist(x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)

