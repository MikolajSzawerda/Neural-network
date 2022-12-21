import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
import time
from collections import namedtuple
from copy import deepcopy

Layer = namedtuple("Layer", "weights biases activ_func deriv_func")


def orig_func(x):
    return np.power(x, 2) * np.sin(x) + 50 * np.sin(2 * x)
    # return np.power(x, 2)


def func(x):
    return 0.005 * orig_func(10.0 * x) + 0.5
    # return np.power(x, 2)


activation = {
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0) * 1),
    'sigmoid': (lambda x: np.divide(1, 1 + np.exp(-x)), lambda x: np.divide(np.exp(x), np.power(1 + np.exp(x), 2))),
    'gaussian': (lambda x: np.exp(-np.power(x, 2)), lambda x: np.multiply(-2 * x, np.exp(-np.power(x, 2)))),
    'linear': (lambda x: x, lambda x: 1)
}


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




def init_network(topology, a_func):
    results = []
    for i, func in enumerate(a_func, 1):
        neuron = (npr.uniform(-1, 1, (topology[i], topology[i - 1])),
                  npr.uniform(-1, 1, (topology[i], 1)),
                  *activation[func])
        results.append(neuron)
    return results


def predict(x, neurons):
    output = x
    for neuron in neurons:
        processed_input = np.matmul(neuron[0], output) + neuron[1]
        output = neuron[2](processed_input)
    return output


def train(X, Y, x_test, y_test, **kwargs):
    neurons = init_network(kwargs['topology'], kwargs['activ_func'])
    mse_rate = []
    epoch = range(kwargs['epoch'])
    batch_size = int(np.ceil(X.shape[0] / kwargs['batch_size']))
    for _ in tqdm(epoch):
        for x_batch, y_batch in zip(np.array_split(X, batch_size), np.array_split(Y, batch_size)):
            x = np.asmatrix(x_batch)
            y = np.asmatrix(y_batch)
            n = x.shape[1]
            rate = -kwargs['learning_rate'] / n
            output = x
            layers = []

            for neuron in neurons:
                processed_input = np.matmul(neuron[0], output) + neuron[1]
                output = neuron[2](processed_input)
                layers.append((processed_input, output))

            back_prop = zip(reversed(layers), reversed(neurons))
            (next_processed_input, next_output), next_neuron = back_prop.__next__()
            delta = np.multiply(next_output - y, next_neuron[3](next_processed_input))
            deltas = [delta]
            for ((processed_input, output), neuron) in back_prop:
                delta = np.multiply(neuron[3](processed_input), next_neuron[0].transpose().dot(delta))
                next_neuron = neuron
                deltas.append(delta)

            prev_output = x
            update_prop = zip(reversed(deltas), layers)
            for i, (delta, (processed_input, output)) in enumerate(update_prop):
                neurons[i][0].__iadd__(rate * delta.dot(prev_output.transpose()))
                neurons[i][1].__iadd__(rate * np.sum(delta, 1))
                prev_output = output

        # y_predict = predict(x_test, neurons)
        # mse_rate.append(np.average(np.power(y_predict - y_test, 2)))
    return mse_rate, neurons


if __name__ == '__main__':
    params = {
        'topology': (1, 20, 20, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 100,
        'batch_size': 20,
        'learning_rate': 0.5,
    }
    x = np.arange(-1, 1, 0.0001)
    np.random.shuffle(x)
    y = np.apply_along_axis(func, 0, x)
    split_index = int(0.7 * x.shape[0])
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    nn = GradientNeuralNetwork(**params)
    nn.train(x_train, y_train)
    # start = time.process_time()
    mse_rate, neurons = train(x_train, y_train, np.asmatrix(x_test), np.asmatrix(y_test), **params)
    # end = time.process_time()
    # print(end - start)
    # pd.DataFrame(mse_rate).plot(logy=True)
    aprox = pd.DataFrame(data=[
        x_test,
        np.asarray(nn.predict(np.asmatrix(x_test))).reshape(-1),
        np.apply_along_axis(func, 0, x_test)
    ]
    ).transpose().sort_values(by=0)
    aprox.rename(columns={0: 'x', 1: 'y_predict', 2: 'y'}, inplace=True)
    aprox.plot('x', ['y', 'y_predict'])
    plt.show()