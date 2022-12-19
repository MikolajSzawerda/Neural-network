import numpy as np
import matplotlib.pyplot as plt
import scipy
from functools import partial
import random


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU(x):
    x[x<0] = 0
    return x


def gaussian(x):
    return np.exp(-(x**2))


def f(x):
    return ((x**2))*np.sin(x) + 100*np.sin(x)*np.cos(x)


def func(x):
    return 0.005*f(10.0*x) + 0.5


def slice_weights(weights, structure, bias_slice_point):
    neuron_layers = list()
    slice_point_weights = 0
    previous_layer = structure[0]
    for layer in structure[1:]:
        neuron_layers.append((
        (np.reshape(weights[slice_point_weights:slice_point_weights+layer*previous_layer], (previous_layer, layer))),
        (weights[bias_slice_point:bias_slice_point+layer]))
        )
        slice_point_weights += layer*previous_layer
        bias_slice_point += layer
        previous_layer = layer
    return neuron_layers


def forward(input, neuron_layers):

    layer_value = input
    for layer_weight, layer_bias in neuron_layers[:-1]:
        layer_value = gaussian(np.matmul(layer_value, layer_weight)+layer_bias)
    return np.dot(layer_value, neuron_layers[-1][0]) + neuron_layers[-1][1]


def calculate_error(weights, structure, bias_slice_point, dataset):
    neuron_layers = slice_weights(weights, structure, bias_slice_point)
    error = 0
    data = np.array([a[0] for a in dataset])
    results = [a[1] for a in dataset]
    predicted_results = forward(data, neuron_layers)
    for predicted_result, result in zip(predicted_results, results):
        error += (predicted_result - result)**2
    return error/len(dataset)


def main():
    structure = (1, 3, 3, 1)
    dataset = [([x], func(x)) for x in [random.uniform(-1.0, 1.0) for _ in range(150)]]
    weights = scipy.optimize.differential_evolution(partial(calculate_error, dataset=dataset, structure=structure, bias_slice_point=15), bounds=[(-10.0, 10.0)]*22, maxiter=100)
    print(weights)
    dataset = [([x], func(x)) for x in [random.uniform(-.0, 1.0) for _ in range(150)]]
    data = np.array([a[0] for a in dataset])
    results = [a[1] for a in dataset]
    network_layers = slice_weights(weights.x, structure, 15)
    prediction_list = forward(data, network_layers)
    for prediction, result in zip(prediction_list, results):
        print(prediction, result)
    for point in dataset:
        plt.plot(point[0], point[1], 'o', color='black')
    for predicted, point in zip(prediction_list, dataset):
        plt.plot(point[0], predicted, 'o', color='red')

    plt.show()




if __name__ == '__main__':
    main()