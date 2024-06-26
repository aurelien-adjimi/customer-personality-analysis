import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        # Randomly initialize centroids
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.labels = np.zeros(X.shape[0])
        self.inertia = 0

        for i in range(self.max_iter):
            # Assign each point to the closest centroid
            self.labels = np.argmin(cdist(X, self.centroids), axis=1)
            # Calculate new centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            # Check if the centroids have converged
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
            self.inertia = np.sum(np.min(cdist(X, self.centroids), axis=1))

    def predict(self, X):
        return np.argmin(cdist(X, self.centroids), axis=1)