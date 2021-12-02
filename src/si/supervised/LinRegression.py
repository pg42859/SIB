import numpy as np
from .Model import Model
from src.si.data.dataset import Dataset
from src.si.util.metrics import mse


class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd
        self.theta = None
        self.num_iterations = epochs
        self.lr = lr

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
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.num_iterations):
            grad = 1/m*(X.dot(self.theta)-Y).dot(X)
            self.theta -= self.lr*grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def train_closed(self, X, Y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, x):
        assert self.is_fited
        _X = np.hstack(([1], x))
        return np.dot(self.theta, _X)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(y_pred, self.Y)


class LinearRegressionReg(LinearRegression):
    def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
        super(LinearRegression).__init__(gd=gd, epochs=epochs, lr=lr)
        self.lbd = lbd

    def train_closed(self, X, Y):
        n = X.shape[1]
        identity = np.eye(n)
        identity[0, 0] = 0  # mudar o primeiro elemento para 0, para nao dar biased
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(Y)
        self.is_fited = True

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

    def predict(self, X):
        assert self.is_fited
        _x = np.hstack(([1], X))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2