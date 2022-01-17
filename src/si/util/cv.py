import numpy as np
import pandas as pd

from si.util.util import train_test_split


class CrossValidationScore:

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get("cv", 3)
        self.score = score
        self.split = kwargs.get("split", 0.8)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.Y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis=1, arr=train.X)
                train_scores.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=1, arr=test.X)
                test_scores.append(self.score(test.Y, y_test))
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores

    def toDataframe(self):
        assert self.train_scores and self.test_scores, "Need to run trainning before hand"
        return np.array((self.train_scores, self.test_scores))


class GridSearchCV:

    def __init__(self, model, dataset, parameters, score=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.score = score
        hasparam = [hasattr(self.model, param) for param in parameters]
        if np.all(hasparam):
            self.parameters = parameters
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f" Wrong parameters: {keys[index]}")
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())
        values = list(self.parameters.values())
        from itertools import product
        for comb in list(product(*values)):
            for attr, value in zip(attrs, comb):
                setattr(self.model, attr, value)
            cv = CrossValidationScore(self.model, self.dataset, self.score, **self.kwargs)
            cv.run()
            self.results.append(cv.run())
        return self.results

    def toDataframe(self):
        assert self.results, "Need to run trainning before hand"
        n_cv = len(self.results[0][0])
        data = np.hstack((np.array([res[0] for res in self.results]), np.array([res[1] for res in self.results])))
        return pd.DataFrame(data=data, columns=[f"CV_{i+1} train" for i in range(n_cv)]+[f"CV_{i+1} test" for i in range(n_cv)])