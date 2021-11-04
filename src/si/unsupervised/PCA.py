import numpy as np
from src.si.util.scale import StandardScaler

class PCA:
    def __init__(self, num_components=2):
        self.nuncomps = num_components

    
    def transform(self,dataset):
        x_scaled = StandardScaler.fit_transform(dataset) #standardizar os dados
        matriz_cov = np.cov(x_scaled, rowvar=False)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(matriz_cov)
        #sort the eigenvalues in descending order
        self.sorted_index = np.argsort(self.eigen_values)[::-1] #dá return a um array de indices do mesmo shape
        self.sorted_eigenvalue = self.eigen_values[self.sorted_index]
        sorted_eigenvectors = self.eigen_vectors[:, 0:self.numcomps]
        eigenvector_subset = sorted_eigenvectors[:, 0:self.nuncomps]
        x_reduced = np.dot(eigenvector_subset.transpose(), x_scaled.transpose()).transpose()
        return np.sum(self.sorted_eigenvalue_sub), self.sorted_eigenvalue_sub
    
    def explained_variance(self, dataset):
        self.sorted_eigenvalue_sub = self.sorted_eigenvalue[0:self.numcomps]
        return np.sum(self.sorted_eigenvalue), self.sorted_eigenvalue_sub
    
    def fit_transform(self, dataset):
        data_reduced = self.transform(dataset)
        explain, eigvalues = self.explained_variance(dataset)
        return data_reduced, explain, eigvalues
