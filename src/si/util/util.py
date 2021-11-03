import itertools
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)
    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        data = np.hstack((dataset.X,dataset.Y.reshape(len(dataset.Y))))
        names= [dataset._xnames,dataset._yname]
    else:
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y))))
        names = [dataset._xnames]
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    maxi = np.max(data, axis=0)
    mini = np.min(data, axis=0)
    stats = {}
    for i in range(data.shape[1]):
        stat = {'mean' : mean[i]
                ,'var' : var[i]
                ,'max' : maxi[i]
                ,'min' : mini[i]}
        stats[names[i]] = stat
    if format == 'df':
        import pandas as pd
        df= pd.DataFrame(stats)
        return df
    else:
        return stats
