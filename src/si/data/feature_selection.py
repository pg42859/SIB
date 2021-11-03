import numpy as np
from copy import copy
import warnings
from ..data import Dataset
import scipy.stats as stats


class VarianceThreshold:
    def __init__(self, threshold=0):
        """
        the variance threshold is a simple baseline approach to feature selection
        it removes all features which variance doesn't meet some threshold limit
        it removes all zero-variance features, i.e..
        """
        if threshold < 0:
            raise Exception('Threshold must be a non negative value')
        else:
            self.threshold = threshold

    def fit(self, dataset):
        X = dataset.X
        self.var = np.var(X, axis=0)

    def transform(self, dataset, inline = False):
        X = dataset.X
        cond = self.var > self.threshold
        ind = []
        for i in range(len(cond)):
            if cond[i]: ind.append(i)
        X_trans = X[:, ind]
        xnames = [dataset._xnames[i] for i in ind]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))

    def fit_transform(self,dataset, inline = False):
        self.fit(dataset)
        return self.transform(dataset, inline)
