import numpy as np
import numpy.random as npr
from tqdm import tqdm
from copy import deepcopy
from utils import activation, Layer, func


class GradientNeuralNetwork:

    @staticmethod
    def init_layers(topology, a_func):
        results = []
        for i, func in enumerate(a_func, 1):
            layer = Layer(npr.uniform(-1, 1, (topology[i], topology[i - 1])),
                          npr.uniform(-1, 1, (topology[i], 1)),
                          *activation[func])
            results.append(layer)
        return results

    def __init__(self, **kwargs):
        self.layers = self.init_layers(kwargs['topology'], kwargs['activ_func'])
        self.learning_rate = kwargs['learning_rate']
        self.epoch = kwargs['epoch']
        self.batch_size = kwargs['batch_size']

    def forward(self, x):
        output = x
        layers_result = []
        for layer in self.layers:
            processed_input = np.matmul(layer.weights, output) + layer.biases
            output = layer.activ_func(processed_input)
            layers_result.append((processed_input, output))
        return layers_result

    def backpropagate(self, y, layers_result):
        back_prop = zip(reversed(layers_result), reversed(self.layers))
        (next_processed_input, next_output), next_layer = back_prop.__next__()
        delta = np.multiply(next_output - y, next_layer.deriv_func(next_processed_input))
        deltas = [delta]
        for ((processed_input, output), layer) in back_prop:
            delta = np.multiply(layer.deriv_func(processed_input),
                                next_layer.weights.transpose().dot(delta))
            next_layer = layer
            deltas.append(delta)
        return deltas

    def improve_weights(self, x, layers_result, deltas):
        prev_output = x
        rate = - self.learning_rate / x.shape[1]
        for layer, results, delta in zip(self.layers, layers_result, deltas):
            layer.weights.__iadd__(rate * delta.dot(prev_output.transpose()))
            layer.biases.__iadd__(rate * np.sum(delta, 1))
            prev_output = results[1]

    def train(self, X, Y):
        batch_size = int(np.ceil(X.shape[0] / self.batch_size))
        neurons_state = []
        for _ in tqdm(range(self.epoch)):
            for x_batch, y_batch in zip(np.array_split(X, batch_size), np.array_split(Y, batch_size)):
                x = np.asmatrix(x_batch)
                y = np.asmatrix(y_batch)
                layers_result = self.forward(x)
                self.improve_weights(x, layers_result, reversed(self.backpropagate(y, layers_result)))
            neurons_state.append(deepcopy(self.layers))
        return neurons_state

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.activ_func(np.matmul(layer.weights, output) + layer.biases)
        return output
