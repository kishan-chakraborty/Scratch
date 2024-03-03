"""
    ML models from scratch series: K MEANS CLASSIFIER

"""
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

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
        self.x_train = None                         # Training data.
        self.n_sample = None                        # no. of trainig data
        self.n_feat = None                          # no. of training features.
        self.centroids = None                       # Initialize the centroids

    def centroid_init(self, method='kmeanspp'):
        """
            Algorithm for better centroid initialization.
            ref: https://en.wikipedia.org/wiki/K-means%2B%2B
        """
        if method is None:      # Random initialization
            init_indices= np.random.choice(self.n_sample, self.k, replace= False)
            self.centroids= [self.x_train[index] for index in init_indices]

        else:                   # KMeans++
            # Find the distance between centroids and data points
            np.random.seed(42)
            idx1 = np.random.choice(range(self.n_sample), size=1, replace=False)
            self.centroids[0] = self.x_train[idx1]
            distances = []

            for i in range(1, self.k):
                # Calculate the distance between last centroid and all the remaining points.
                distance =  ((self.x_train - self.centroids[i-1]) *
                            (self.x_train - self.centroids[i-1])).sum(axis=1)
                distances.append(distance)
                # Find the minimum distance among all the centroids.
                distance2 = np.min(distances, axis=0)
                # Normalise the calculated distances to form probability distribution.
                prob = distance2 / distance2.sum()
                # Find the next centroid based on the calculated prob distribution.
                centroid_next = np.random.choice(range(self.n_sample), size=1, p=prob)
                self.centroids[i] = self.x_train[centroid_next]

    def train(self, x_train: np.array):
        """
            The objective is to find k centroids based on provided training data.
            x_train: Training data
        """
        self.x_train = x_train
        self.n_feat = self.x_train.shape[1]            # Feature Dimension.
        self.n_sample = self.x_train.shape[0]          # No of samples.
        self.centroids = np.zeros((self.k, self.n_feat))
        self.centroid_init()    # Initialize centroids using kmeans++ algorithm or randomly.

        for _ in range(self.max_iter):
            cluster_labels = self._assign_cluster_labels(self.x_train, self.centroids)
            centroids_old= self.centroids
            self._update_centroids(cluster_labels)

            if self._is_converged(self.centroids, centroids_old):
                break

    def predict(self, x_test: np.array) -> np.array:
        """
            Assign each test example an cluster based on its distance with the centroids.
            x_test: Test data of shape [, self.n_feat].
        """
        # Calculate the distance of each example from the centroids.
        label_assigned = self._assign_cluster_labels(x_test, self.centroids)
        return label_assigned

    def _assign_cluster_labels(self, samples: np.array, centroids: np.array) -> np.array:
        """
            To assign cluster labels to every sample.
            samples: Data to assign cluster labels.
            centroids: numpy array containing the current centroids. shape: [k, samples.shape[1]]
        """
        # Reshape the centroid array to parallelize the distance calculation.
        centroids3d = centroids.reshape(self.k, 1, self.n_feat)# shape: [k,1, X.shape[1]]
        distance = ((samples-centroids3d)**2).sum(axis=-1)       # shape: [k, X.shape[1]]
        out = distance.argmin(axis=0)                            # shape: [x.shape[1]]
        return out

    def _update_centroids(self, cluster_labels: np.array) -> np.array:
        """
            cluster_labels: array of shape (self.n_sample), i'th value represents the cluster
                            label corresponding to i'th training sample.
            returns: the updaated centroid values.
        """
        for i in range(self.k):
            # Index of the samples beloning to ith cluster.
            idx_i = cluster_labels[cluster_labels == i]
            # Updating the centroids based on the mean value of sample belonging to ith cluster.
            self.centroids[i] = self.x_train[idx_i].mean(axis=0)

    def _is_converged(self, old_centroids, new_centroids):
        """
            To check if the training is complete. We consider the training process is complete when
            there is no significant difference between old and updated centroid values.
        """
        distance = (old_centroids - new_centroids)**2
        out = (distance <= 1e-10).all()
        return out

if __name__ == "__main__":
    # Generate data
    X, y= make_blobs(centers= 4, n_samples= 500, n_features= 2, shuffle= True, random_state= 42)
    model = KMeans(4, 100)
    model.train(X)
    predicted_class = model.predict(X)
    plt.scatter(X[predicted_class== 2][:, 0], X[predicted_class== 2][:, 1], color= 'r')
    plt.scatter(X[predicted_class== 1][:, 0], X[predicted_class== 1][:, 1], color= 'k')
    plt.scatter(X[predicted_class== 0][:, 0], X[predicted_class== 0][:, 1], color= 'g')
    plt.scatter(X[predicted_class== 3][:, 0], X[predicted_class== 3][:, 1], color= 'b')
    plt.show()