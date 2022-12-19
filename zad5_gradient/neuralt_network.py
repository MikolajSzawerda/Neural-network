import numpy as np
import pandas as pd
from collections import namedtuple
from copy import deepcopy
import matplotlib.pyplot as plt

Neuron = namedtuple("Neuron", "weights bias")


class NeuralNetwork:
    def __init__(self):
        self.neurons = self.init_neurons(2)
        self.ac_func = lambda x: np.maximum(0, x)
        self.ac_deriv = lambda x: np.array(list(map(lambda y: [int(y > 0)], x)))


    @staticmethod
    def init_neurons(n):
        return [Neuron(np.ones((n, 1)), np.ones((n, 1))),
               Neuron(np.ones((n, 2)), np.ones((n, 1))),
                Neuron(np.ones((1, n)), np.zeros(1))]

    def forward_propagation(self, x):
        a = np.asmatrix(x)
        results = []
        for neuron in self.neurons:
            z = neuron.weights.dot(a)+neuron.bias
            a = self.ac_func(z)
            results.append((z, a))
        return results

    def backward_propagation(self, forward: list, y):
        layers = zip(reversed(self.neurons), reversed(forward))
        neuron, (z, a) = layers.__next__()
        delta = np.multiply(a-y, self.ac_deriv(z))
        deltas = [delta]
        for new_neuron, (new_z, new_a) in layers:
            delta = np.multiply(neuron.weights.transpose()*delta, self.ac_deriv(new_z))
            neuron, z = new_neuron, new_z
            deltas.append(delta)
        return reversed(deltas)

    def predict(self, x):
        a = np.asmatrix(x)
        for neuron in self.neurons:
            z = neuron.weights.dot(a)+neuron.bias
            a = self.ac_func(z)
        return a[0, 0]

    def train(self, X, Y):
        l_r = 0.5
        layer_data = self.forward_propagation(X)
        deltas = self.backward_propagation(layer_data, Y)
        old_a = X
        results = []
        for i, ((z, a), delta) in enumerate(zip(layer_data, deltas)):
            new_w = np.multiply(delta, old_a)
            new_b = delta
            results.append((new_w, new_b))
            old_a = a
        return results

    def mini_batch(self, data):
        updates = []
        l_r = 1.0
        for x, y in data:
            updates.append(self.train(x, y))
        for i, layer in enumerate(zip(*updates)):
            m = len(layer)
            neuron = self.neurons[i]
            new_w = neuron.weights - (l_r/m)*sum(x[0] for x in layer)
            new_b = neuron.bias - (l_r/m)*sum(x[1] for x in layer)
            self.neurons[i] = Neuron(new_w, new_b)



def func(x):
    return np.power(x, 2)*np.sin(x)+50*np.sin(2*x)


if __name__ == '__main__':
   nn = NeuralNetwork()
   mse_y = []
   for i in range(100):
       data = [(x, func(x)) for x in np.arange(-10, 10, 0.2)]
       nn.mini_batch(data)
       mse = np.average([(nn.predict(x)-y)**2 for x, y in data])
       mse_y.append(mse)
   plt.plot(range(100), mse_y)
   plt.show()
