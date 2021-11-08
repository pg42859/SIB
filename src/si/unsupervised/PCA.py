import numpy as np
from src.si.util.scale import StandardScaler
import pandas as pd

class PCA:
    def __init__(self, num_components=2, using="svd"):
        self.numcomps = num_components
        self.alg = using

    def transform(self, dataset):  # objeto Dataset
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(dataset)  
        X = x_scaled.X
        XT = X.transpose()
        print(XT)
        self.eigen_vectors, self.eigen_values, vt = np.linalg.svd(XT)
        self.sorted_index = np.argsort(self.eigen_values)[::-1]  
        self.sorted_eigenvalue = self.eigen_values[self.sorted_index] 
        sorted_eigenvectors = self.eigen_vectors[:, self.sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, 0:self.numcomps] 
        x_reduced = np.dot(eigenvector_subset.transpose(), XT).transpose()
        return x_reduced

    def variance_explained(self):
        somapercent = np.sum(self.sorted_eigenvalue)
        percentagem = []
        for i in self.sorted_eigenvalue:
            percentagem.append(i/somapercent *100)
        return np.array(percentagem)

    def fit_transform(self, dataset):
        data_reduced = self.transform(dataset)
        percentagem = self.variance_explained()
        return data_reduced, percentagem