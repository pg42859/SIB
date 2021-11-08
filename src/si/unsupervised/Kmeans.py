from src.si.util.util import euclidean, manhattan
from src.si. data import Dataset
import numpy as np

class KMeans:
    def __init__(self, k: int, max_iterations=100, measure="euclidean"):
        self.k = k
        self.n = max_iterations
        self.centroides = None
        if measure is "euclidean":
            self.measure = euclidean

    def fit(self, dataset):
        self.min = np.min(dataset.X, axis=0)  # fazer a média em relação às features
        self.max = np.max(dataset.X, axis=0)

    def init_centroides(self, dataset):
        x = dataset.X
        self.centroides = np.array([np.random.uniform(low=self.min[i], high=self.max[i], size=(self.k,))
                                   for i in range(x.shape[1])]).T

    def get_closest_centroid(self, x):
        dist = self.measure(x, self.centroides)
        closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroides(dataset)
        x = dataset.X
        changed = False
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while count < self.n and not changed:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.T)
            self.centroids = np.array([np.mean(x[idxs == i], axis=0) for i in range(self.k)])
            changed = np.all(old_idxs == idxs)
            old_idxs = idxs
            count += 1
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        centroides, indices = self.transform(dataset)
        return centroides, indices


