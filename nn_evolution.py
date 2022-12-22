import numpy as np
import scipy
from functools import partial
from collections import namedtuple
from utils import Layer, activation, func

MatrixView = namedtuple("MatrixView", "w_a w_b w_col w_row b_a b_b")


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def gaussian(x):
    return np.exp(-np.power(x, 2))


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

    @staticmethod
    def to_common_form(results, activ_funcs):
        return [
            Layer(weights.transpose(),
                  np.asmatrix(biases).transpose(),
                  *activation[func])
            for ((weights, biases), func) in zip(results, activ_funcs)
        ]

    def __init__(self, **kwargs):
        self.structure = kwargs['topology']
        self.bias_slice_point = self.calculate_bias_slice_point(self.structure)
        self.slice_points = self.calculate_slice_points(self.structure, self.bias_slice_point)
        self.epoch = kwargs['epoch']
        self.neurons_state = []
        self.weights = []
        self.activ_funcs = kwargs['activ_func']

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

    def forward(self, weights, input):
        layer_value = input
        neuron_layers = self.get_matrix_form(weights)
        for layer_weight, layer_bias in neuron_layers[:-1]:
            layer_value = gaussian(np.matmul(layer_value, layer_weight) + layer_bias)
        return np.dot(layer_value, neuron_layers[-1][0]) + neuron_layers[-1][1]

    def evaluate(self, weights, X, Y):
        predicted_results = self.forward(weights, X)
        return np.average(np.power(predicted_results - Y, 2))

    def save_neuron_state(self, network_state, **kwargs):
        self.neurons_state.append(network_state)
        return False

    def train(self, X, Y):
        n_of_parameters = self.bias_slice_point + sum(self.structure[1:])
        results = \
            scipy.optimize.differential_evolution(partial(self.evaluate,
                                                          X=np.asmatrix(X).transpose(),
                                                          Y=np.asmatrix(Y).transpose()),
                                                  popsize=1,
                                                  bounds=[(-10.0, 10.0)] * n_of_parameters,
                                                  maxiter=self.epoch,
                                                  callback=self.save_neuron_state,
                                                  # disp=True,
                                                  tol=1e-5,
                                                  mutation=(0.0, 1.99),
                                                  recombination=0.5)
        self.weights = self.to_common_form(self.get_matrix_form(results.x), self.activ_funcs)
        results = []
        for network_state in self.neurons_state:
            results.append(self.to_common_form(self.get_matrix_form(network_state), self.activ_funcs))
        return results

    def predict(self, x):
        output = np.asmatrix(x)
        for layer in self.weights:
            output = layer.activ_func(np.matmul(layer.weights, output) + layer.biases)
        return output
