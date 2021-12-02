from src.si.util.util import euclidean, manhattan
from src.si. data import Dataset
import numpy as np

class KMeans:
    def __init__(self, k: int, max_iterations=100, measure="euclidean"):
        self.k = k
        self.n = max_iterations  # max de iterações
        self.centroides = None
        if measure is "euclidean":
            self.measure = euclidean

    def fit(self, dataset):
        self.min = np.min(dataset.X, axis=0)  # minimo
        self.max = np.max(dataset.X, axis=0)  # maxim0

    def init_centroides(self, dataset):  # descobre todos os centroides, transforma em array e faz a transposta
        x = dataset.X
        self.centroides = np.array([np.random.uniform(low=self.min[i], high=self.max[i], size=(self.k,))
                                   for i in range(x.shape[1])]).T
        # low: valores gerados vão ser maiores ou iguais - lower bound
        # high: valores gerados vão ser menors ou iguais - upper bound
        # corre todas as colunas

    def get_closest_centroid(self, x):  # dá return do index do centroide mais próximo
        dist = self.measure(x, self.centroides)  # calcula a distancia
        closest_centroid_index = np.argmin(dist, axis=0)  # argmin - recebe um array e o axis pretendido, dá return dos indices de menor valor ao longo desse axis (coluna neste caso)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroides(dataset)
        x = dataset.X
        changed = False
        count = 0
        old_idxs = np.zeros(x.shape[0])  # array de zeros
        while count < self.n and not changed:  # enquanto o nº max de its não for atingido e changed = False
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.T)  # aplica ao longo do axis escolhido ao array x.T a função get_closest_centroid
            self.centroids = np.array([np.mean(x[idxs == i], axis=0) for i in range(self.k)])  # calcular a media sobre os pontos e essas medias são os novos pontos
            changed = np.all(old_idxs == idxs)  # testar se são os mesmo indexes que os antigos
            old_idxs = idxs  # indexes antigos passam a ser os novos
            count += 1  # aumentar o nº da iteração
        return self.centroids, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        centroides, indices = self.transform(dataset)
        return centroides, indices


