import numpy as np
from .Model import Model
from si.util.util import euclidean
from si.util.metrics import accuracy_score


class KNN(Model):
    """
    K Vizinhos Mais Próximos (método lento na fase de previsão)
    """

    def __init__(self, number_neighboors, classification=True):
        super(KNN).__init__()
        self.number_neighboors = number_neighboors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighboors(self, x):

        """
        Calcula as distâncias entre cada ponto de teste
        em relação a todos os pontos do dataset de treino
        """

        distance = euclidean(x,
                             self.dataset.X)  # calcular a distância entre um ponto e todos os outros (distância euclidiana)
        idxs_sort = np.argsort(distance)  # dá sort aos idxs tendo em conta a distância
        return idxs_sort[:self.number_neighboors]  # retorna os idxs dos melhores pontos (vizinhos mais próximos)

    def predict(self, x):

        """
        :param x: array de teste
        :return: predicted labels
        """

        assert self.is_fitted, 'Model must be fot before prediction'
        viz = self.get_neighboors(x)  # pontos mais próximos de x
        values = self.dataset.Y[viz].tolist()  # transforma os valores em lista
        if self.classification:  # se for uma variável discreta/fatorial
            prediction = max(set(values),
                             key=values.count)  # avalia os valores máximos -> CLASSIFICAÇÃO (devolve a classe com mais incidência)
        else:  # se for uma variável contínua/numérica
            prediction = sum(values) / len(values)  # média dos valores -> REGRRESSÃO
        return prediction

    def cost(self):
        y_pred = np.ma.apply_along_axis(self.predict, axis=0,
                                        arr=self.dataset.X.T)  # ma: máscara; temos de usar se não vai formatar a previsão
        return accuracy_score(self.dataset.Y, y_pred)  # dá a precisão