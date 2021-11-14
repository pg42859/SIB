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

    def fit(self, dataset):  # calcula a variancia
        X = dataset.X  # variaveis nao dependentes
        self.var = np.var(X, axis=0)  # aplica a variancia por linahs e guarda em self.var

    def transform(self, dataset, inline = False):  # escolhe as variaveis que são maiores do que o threshold
        X = dataset.X
        cond = self.var > self.threshold  #  array de booleanos
        ind = []
        for i in range(len(cond)):
            if cond[i]:
                ind.append(i)  # se a cond for verdadeira, vai dar append do indice dessa condição
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

    def fit(self, dataset):  # calcular o fscore e o pvalue
        self.fscore, self.pvalue = self.function(dataset)

    def transform(self, dataset, inline=False):
        X = copy(dataset.X) # valores de x
        xnames = copy(dataset._xnames)
        sel_list = np.argsort(self.fscore)[-self.feat_num:]
        # np.argsort(self.fscore) - retorna indices ordenados de acordo com o fscore
        # [-self.feat_num:] - vai buscar os ultimos indices, uma vez que queremos os scores mais altos
        featdata = X[:, sel_list] # selecionar as features
        featnames = [xnames[index] for index in sel_list]  # vai buscar os nomes através do indice
        if inline: # se for true, faz a alteração no dataset
            dataset.X = featdata
            dataset._xnames = featnames
            return dataset
        else: # se for false, cria um dataset novo
            return Dataset(featdata, copy(dataset.Y), featnames, copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):  # fit to data, then transform it
        self.fit(dataset)
        return self.transform(dataset, inline=inline)


def f_regress(dataset):  # testa a hipotese nula de que 2 ou mais grupos tem a mesma população média
    X, y = dataset.getXy()
    args = []
    for k in np.unique(y):  # valores unicos em y
        args.append(X[y == k, :])
    from scipy.stats import f_oneway
    F_stat, pvalue = f_oneway(*args)
    return F_stat, pvalue