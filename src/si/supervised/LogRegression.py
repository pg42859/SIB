from .Model import Model
import numpy as np
from ..util.util import sigmoide, add_intersect


class LogisticRegression(Model):
    def __init__(self, epochs=1000, lr=0.001):
        super(LogisticRegression, self).__init__()
        self.theta = None
        self.epochs = epochs
        self.lr = lr

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        self.train_gd(X, Y)
        self.is_fited = True

    def train_gd(self, X, Y):
        n = self.X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        for epoch in range(self.epochs):
            Z = np.dot(self.X, self.theta)
            h = sigmoide(Z)
            gradient = np.dot(self.X.T, (h - self.Y)) / self.Y.size
            self.theta -= self.lr * gradient
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, x):
        if not self.is_fited:
            raise Exception("The model hasn't been fitted yet.")
        _x = np.hstack(([1], x))
        return np.round(sigmoide(np.dot(self.theta, _x)))

    def cost(self, X=None, Y=None, theta=None):
        if X is not None:
            X = add_intersect(X)
        else:
            X = self.X
        if Y is not None:
            Y = Y
        else:
            Y = self.Y
        if theta is not None:
            theta = theta
        else:
            theta = self.theta
        y_pred = np.dot(X, theta)
        h = sigmoide(y_pred)
        cost = (Y * np.log(h) + (1-Y) * np.log(1-h))
        res = -np.sum(cost) / X.shape[0]
        return res


class LogisticRegressionReg(LogisticRegression):
    def __init__(self, epochs = 1000, lr=0.1, lbd=1):
        super(LogisticRegressionReg, self).__init__()
        self.lbd = lbd
        self.epochs = epochs
        self.lr = lr

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.epochs):
            Z = np.dot(self.X, self.theta)
            h = sigmoide(Z)
            grad = np.dot(self.X.T, (h-self.Y)) / self.Y.size
            grad[1:] = grad[1:] + (self.lbd/m) * self.theta[1:]
            self.theta -= self.lr * grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fited, 'model must be fitted before predicting'
        _x = np.hstack(([1], X))
        p = sigmoide(np.dot(self.theta, _x))
        if p <= 0.5:
            return 0
        else:
            return 1

    def cost(self, X=None, Y=None, theta=None):
        if X is not None:
            X = add_intersect(X)
        else:
            X = self.X
        if Y is not None:
            Y = Y
        else:
            Y = self.Y
        if theta is not None:
            theta = theta
        else:
            theta = self.theta
        m = X.shape[0]
        h = sigmoide(np.dot(X, theta))
        cost = (Y * np.log(h) + (1-Y) * np.log(1-h))
        reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2*m)
        res = (-np.sum(cost)/m)+reg
        return res