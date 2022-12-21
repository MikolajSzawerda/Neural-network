import numpy as np
from functools import partial
from nn_gradient import GradientNeuralNetwork
from nn_evolution import EvolutionNeuralNetwork


class NeuralNetwork:
    def __init__(self, **kwargs):
        if kwargs['method'] == 'gradient':
            self.network = GradientNeuralNetwork(**kwargs)
        else:
            self.network = EvolutionNeuralNetwork(**kwargs)
        self.results = []

    def train(self, X, Y):
        self.results = self.network.train(X, Y)

    def predict(self, X):
        return self.network.predict(X)

    @staticmethod
    def calculate_mse(layers, x_test, y_test):
        output = np.asmatrix(x_test)
        for layer in layers:
            output = layer.activ_func(np.matmul(layer.weights, output) + layer.biases)
        return np.average(np.power(y_test - output, 2))

    def get_mse_progress(self, x_test, y_test):
        return list(map(lambda x: self.calculate_mse(x, x_test, y_test), self.results))

