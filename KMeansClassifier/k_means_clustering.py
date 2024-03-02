"""
    ML models from scratch series: K MEANS CLASSIFIER

"""
import numpy as np


class KMeans:
    """
        Implementing k means classification algorithm using numpy.
    """
    def __init__(self, k: int, max_iter: int):
        """
            k: No. of neighbouring elements to consider.
            max_iter: Maximum iteration allowed for model convergence.
        """
        self.k = k
        self.max_iter = max_iter
        self.x_train = None         # Training data. [Since this is a lazy type model there is no model training]

        #Keep track of centroids for the clusters
        self.centroids= []

        #Keep track of indices of samples in each clusters
        self.clusters= [[] for _ in range(self.k)]

    def train(self, x_train):

    def predict(self, X):
        self.X= X
        self.n_sample, self.n_features= X.shape

        init_indices= np.random.choice(self.n_sample, self.k, replace= False)
        self.centroids= [self.X[index] for index in init_indices]

        for _ in range(self.max_iter):
            self.clusters= self._create_clusters(self.centroids)

            centroids_old= self.centroids
            self.centroids= self._get_centroids(self.clusters)

            if self._is_converged(self.centroids, centroids_old):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        self.labels= np.empty(self.n_sample)
        for cluster_idx, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels[idx]= cluster_idx
        
        return self.labels


    def _create_clusters(self, centroids):
        clusters= [[] for i in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx= self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters


    def _closest_centroid(self, sample, centroids):
        return np.argmin([Euclidean(sample, centroids[i]) for i in range(self.k)])

    def _get_centroids(self, clusters):
        centroids= np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean= np.mean(self.X[cluster], axis= 0)
            centroids[cluster_idx]= cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        distance= [Euclidean(old_centroids[i], new_centroids[i]) for i in range(self.k)]
        return np.sum(distance)== 0