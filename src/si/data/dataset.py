import numpy as np
from src.si.util.util import label_gen

__all__ = ['Dataset', 'summary']


class Dataset:
    def __init__(self, X=None, Y=None, xnames: list = None, yname: str = None):
        """ Tabular Dataset"""
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X
        self.Y = Y
        self._xnames = xnames if xnames else label_gen(X.shape[1])
        self._yname = yname if yname else 'Y'

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a DataSet from a data file.

        :param labeled:
        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :return: A DataSet object
        :rtype: DataSet
        """
        data = np.genfromtxt(filename, delimiter=sep) #dá return a uma matriz numpy
        if labeled: #se houver labels
            X = data[:, 0:-1] #é tudo extraido menos uma coluna
            Y = data[:, -1] #ultima coluna são os labels
        else: #se não tiver
            X = data
            Y = None #não temos labels
        return cls(X, Y)

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a DataSet in array form from a pandas dataframe.

        :param df: pandas dataframe
        :type df: Dataframe
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: DataSet in array form
        :rtype: array
        """
        if ylabel and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel].to_numpy()
            y = df.loc[:, ylabel].to_numpy()
            xnames = list(df.columns)
            xnames.remove(ylabel)
            yname = ylabel
        else:
            X = df.to_numpy()
            y = None
            xnames = list(df.columns)
            yname = None
        return cls(X, y, xnames, yname)


    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return self.Y is not None

    def getNumFeatures(self):
        """Returns the number of features (numero de colunas de X)"""
        return self.X.shape[1]


    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        unique = np.unique(self.Y, return_counts=False)
        return len(unique) if self.hasLabel() else 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file
        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """
        fullds = np.hstack((self.X, self.Y.reshape(len(self.Y), 1)))
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        import pandas as pd
        if self.Y is None:
            dataset = pd.DataFrame(self.X.copy(), columns=self._xnames[:])
        else:
            dataset = pd.DataFrame(np.hstack((self.X, self.Y.reshape(len(self.Y), 1))), columns=np.hstack((self._xnames, self._yname)))
        return dataset

    def getXy(self):
        return self.X, self.Y


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)
    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        fullds = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y), 1)))
        columns = dataset._xnames[:] + [dataset._yname]
    else:
        fullds = dataset.X
        columns = dataset._xnames[:]
    stats = {}
    for i in range(fullds.shape[1]):
        try:
            _means = np.mean(fullds[:, i], axis=0)
            _vars = np.var(fullds[:, i], axis=0)
            _maxs = np.max(fullds[:, i], axis=0)
            _mins = np.min(fullds[:, i], axis=0)
        except Exception:
            _means = _vars = _maxs = _mins = np.NAN
        stat = {'mean': _means,
                'var': _vars,
                'min': _mins,
                'max': _maxs
                }
        stats[columns[i]] = stat
    if format == 'df':
        import pandas as pd
        df = pd.DataFrame(stats)
        return df
    else:
        return stats

