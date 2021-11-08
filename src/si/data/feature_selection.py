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
        self.var = None
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
            if cond[i]:
                ind.append(i)
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

class SelectKBest:
    def __init__(self, k, funcao_score="f_regress"):
        self.feat_num = k
        if funcao_score == "f_regress":
            self.function = f_regress
        self.fscore = None
        self.pvalue = None

    def fit(self, dataset):
        self.fscore, self.pvalue = self.function(dataset)

    def transform(self, dataset, inline=False):
        X = copy(dataset.X)
        xnames = copy(dataset._xnames)
        sel_list = np.argsort(self.fscore)[-self.feat_num:]
        featdata = X[:, sel_list]
        featnames = [xnames[index] for index in sel_list]
        if inline:
            dataset.X = featdata
            dataset._xnames = featnames
            return dataset
        else:
            return Dataset(featdata, copy(dataset.Y), featnames, copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_regress(dataset):
    X, y = dataset.getXy()
    args = []
    for k in np.unique(y):
        args.append(X[y == k, :])
    from scipy.stats import f_oneway
    F_stat, pvalue = f_oneway(*args)
    return F_stat, pvalue