import numpy as np
import matplotlib.pyplot as plt
import scipy
from functools import partial
import random
from collections import namedtuple
import time

MatrixView = namedtuple("MatrixView", "w_a w_b w_col w_row b_a b_b")
Layer = namedtuple("Layer", "weights biases")


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def gaussian(x):
    return np.exp(-np.power(x, 2))


def f(x):
    return np.power(x, 2) * np.sin(x) + 100 * np.sin(x) * np.cos(x)


def func(x):
    return 0.005 * f(10.0 * x) + 0.5


class EvolutionNeuralNetwork:

    @staticmethod
    def calculate_bias_slice_point(structure):
        bias_slice_point = 0
        previous = structure[0]
        for layer in structure[1:]:
            bias_slice_point += previous * layer
            previous = layer
        return bias_slice_point

    @staticmethod
    def calculate_slice_points(structure, bias_slice_point):
        slice_point_weights = 0
        previous_layer = structure[0]
        slice_points = []
        for layer in structure[1:]:
            slice_points.append(MatrixView(slice_point_weights,
                                           slice_point_weights + layer * previous_layer,
                                           previous_layer,
                                           layer,
                                           bias_slice_point,
                                           bias_slice_point + layer))
            slice_point_weights += layer * previous_layer
            bias_slice_point += layer
            previous_layer = layer
        return slice_points

    def __init__(self, structure):
        self.structure = structure
        self.weights = []
        self.bias_slice_point = self.calculate_bias_slice_point(structure)
        self.slice_points = self.calculate_slice_points(structure, self.bias_slice_point)

    def get_matrix_form(self, weights):
        neuron_layers = list()
        for layer in self.slice_points:
            neuron_layers.append(
                (
                    np.reshape(weights[layer.w_a:layer.w_b], (layer.w_col, layer.w_row)),
                    weights[layer.b_a:layer.b_b]
                )
            )
        return neuron_layers

    def predict(self, weights, input):
        layer_value = input
        neuron_layers = self.get_matrix_form(weights)
        for layer_weight, layer_bias in neuron_layers[:-1]:
            layer_value = gaussian(np.matmul(layer_value, layer_weight) + layer_bias)
        return np.dot(layer_value, neuron_layers[-1][0]) + neuron_layers[-1][1]

    def evaluate(self, weights, X, Y):
        predicted_results = self.predict(weights, X)
        return np.average(np.power(predicted_results - Y, 2))

    def fit(self, dataset):
        n_of_parameters = self.bias_slice_point + sum(self.structure[1:])
        weights = \
            scipy.optimize.differential_evolution(partial(self.evaluate,
                                                          X=np.asmatrix(dataset[:, 0]).transpose(),
                                                          Y=np.asmatrix(dataset[:, 1]).transpose()),
                                                  popsize=1,
                                                  bounds=[(-10.0, 10.0)] * n_of_parameters,
                                                  maxiter=3000,
                                                  # disp=True,
                                                  tol=1e-5,
                                                  mutation=(0.0, 1.99),
                                                  recombination=0.5)
        return weights.x


def main():
    structure = (1, 6, 6, 1)
    x = np.linspace(-1.0, 1.0, 500)
    y = func(x)
    dataset = np.column_stack((x, y))
    ev = EvolutionNeuralNetwork(structure)
    weights = ev.fit(dataset)
    dataset = [([x], [func(x)]) for x in
               [random.uniform(-1.0, 1.0) for _ in range(100)]]
    data = np.array([a[0] for a in dataset])
    results = [a[1] for a in dataset]
    start = time.time()
    prediction_list = ev.predict(weights, data)
    end = time.time()
    print(end-start)
    # for prediction, result in zip(prediction_list, results):
    #     print(prediction, result)
    for point in dataset:
        plt.plot(point[0], point[1], 'o', color='black')
    for predicted, point in zip(prediction_list, dataset):
        plt.plot(point[0], predicted, 'o', color='red')

    # plt.savefig(f'{random.random}')
    plt.show()


if __name__ == '__main__':
    main()
