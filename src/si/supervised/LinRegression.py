import numpy as np
from .Model import Model
from src.si.data.dataset import Dataset
from src.si.util.metrics import mse


class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd  # gradient descendent
        self.theta = None
        self.num_iterations = epochs
        self.lr = lr  # learning rate, velocidade de atualização de parametros

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        if self.gd:  # se quiser com gradiente
            self.train_gd(X, Y)
        else:
            self.train_closed(X, Y) # se quiser étodo dos mins quadrados
        self.is_fited = True

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}  # vai guardar o resultado das thetas e dos erros (custos) a cada iteração
        self.theta = np.zeros(n)  # matriz de zeros com o tamanho do dataset (numero de linhas)
        for epoch in range(self.num_iterations):  # para cada it
            grad = 1/m*(X.dot(self.theta)-Y).dot(X)  # função diferenciavel
            self.theta -= self.lr*grad  # atualização do theta a cada iteração
            self.history[epoch] = [self.theta[:], self.cost()]  # guarda esta iteração e os seus resultados

    def train_closed(self, X, Y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)  # metodo dos minimos quadrados e que minimiza a função erro

    def predict(self, x):  # faz as previsões
        assert self.is_fited
        _X = np.hstack(([1], x))
        return np.dot(self.theta, _X)

    def cost(self):  # calcula o erro
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2


class LinearRegressionReg(LinearRegression):  # para sobreajustamento
    def __init__(self, gd=False, epochs=1000, lbd=1):
        super(LinearRegressionReg, self).__init__()
        self.lbd = lbd  # parametro de regularização
        self.gd = gd
        self.num_iterations = epochs

    def train_closed(self, X, Y):
        n = X.shape[1]
        identity = np.eye(n)  # matriz identidade
        identity[0, 0] = 0  # mudar o primeiro elemento para 0, para nao dar biased
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(Y)  # método analitico
        self.is_fited = True

    def train_gd(self, X, Y):
        m = X.shape[0]  # nº linhas
        n = X.shape[1]  # nº colunas
        self.history = {}
        self.theta = np.zeros(n)  # matriz de zeros com o tamanho do dataset
        lbds = np.full(m, self.lbd)  # matriz quadradada com o nº linhas do dataset e com os valores de regularização
        lbds[0] = 0
        for epoch in range(self.num_iterations):
            grad = (X.dot(self.theta)-Y).dot(X)  # metodo do gradiente
            self.theta -= (self.lr/m)*(lbds+grad)  # atualiza os valores de theta
            self.history[epoch] = [self.theta[:], self.cost()]  # guarda atualizações e erros de cada iteração

    def predict(self, X):
        assert self.is_fited
        _x = np.hstack(([1], X))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2