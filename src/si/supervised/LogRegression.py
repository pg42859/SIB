from .Model import Model
import numpy as np
from ..util.util import sigmoide
from ..util.metrics import mse

class LogisticRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LogisticRegression, self).__init__()

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        if self.gd:
            self.train_gd(X, Y)
        else:
            self.train_closed(X, Y)
        self.is_fited = True

    def train_gd(self, X, Y):
        pass

    def predict(self, x):
        assert self.is_fited
        _X = np.hstack(([1], x))
        return np.dot(self.theta, _X)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(y_pred, self.Y)


class LogisticRegressionReg(LogisticRegression):
    def __init__(self):
        super(LogisticRegressionReg).__init__()

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.num_iterations):
            grad = (X.dot(self.theta)-Y).dot(X)
            self.theta -= (self.lr/m)*(lbds+grad)
            self.history[epoch] = [self.theta[:], self.cost()]