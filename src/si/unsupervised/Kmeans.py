# k = z
# gerar obrigatoriamente k centroides
# Enquanto os clustaers mudam ou até ser atingido o numero máximo de iterações
#   distancia de cada ponto a cada centroide
#   identificar para cada ponto o centroide mais proximo
#   definir os clusters e calcular os novos centroides

# 1º - ou centralizar: X.np.mean(k,) ou usar o standardScaler

from numpy.testing._private.utils import measure
from src.si.util.util import euclidean, manhattan
from src.si. data import Dataset
import pandas as np


class KMeans:
    def __init__(self, k: int, max_iterations=100):
        self.k = k
        self.n = max_iterations
        self.centroides = None
        self.measure = measure
        self._min = 0
        self._max = 0

  #  def distance_12(self, x, y):
        """
        Distancia euclidian distance
        :param x:
        :param y:
        :return:
        """
   #     dist = np.absolute((x - y) ** 2).sum(axis=1)
   #     return dist

    def fit(self, dataset):
        self._min = np.min(dataset.X)
        self._max = np.max(dataset.X)

    def init_centroids(self, dataset):
        x = dataset.X
        self.centroides = np.array([np.random.uniform(low=self._min[1], high=self._max[1], size=self.k)
                                    for i in range(x.shape[1])])

    def get_closest_centroid(self, x):
        if self.measure is "euclidean":
            dist = euclidean(x, self.centroides)
            closest_centroid_index = np.argmin(dist, axis=0)
        else:
            dist = manhattan(x, self.centroides)
            closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        x = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while changed is True or count < self.n:
            idxs = np.apply_along_axis(self.get_closest_centroid(x), axis=0, arr=True)
            cent = [np.mean(x[idxs == 1]) for i in range(x.shape[0])]
            self.centroides = np.array(cent)

            old_idxs = idxs
            count += 1
        return self.centroides, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        centroides, indices = self.transform(dataset)
        return centroides, indices


