import numpy as np
import matplotlib.pyplot as plt
import scipy
from functools import partial
import random


def sigmoid(x):
    return 1/(1+np.exp(-x))


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
            (np.reshape(weights[slice_point_weights:slice_point_weights
                                + layer*previous_layer],
                        (previous_layer, layer))),
            (weights[bias_slice_point:bias_slice_point+layer]))
        )
        slice_point_weights += layer*previous_layer
        bias_slice_point += layer
        previous_layer = layer
    return neuron_layers


def predict(weights, structure, input):
    bias_slice_point = 0
    previous = structure[0]
    for layer in structure[1:]:
        bias_slice_point += previous*layer
        previous = layer
    layer_value = input
    neuron_layers = slice_weights(weights, structure, bias_slice_point)
    for layer_weight, layer_bias in neuron_layers[:-1]:
        layer_value = gaussian(np.matmul(layer_value, layer_weight)+layer_bias)
    return np.dot(layer_value, neuron_layers[-1][0]) + neuron_layers[-1][1]


def evaluate(weights, structure, dataset):
    data = np.array([a[0] for a in dataset])
    results = np.array([a[1] for a in dataset])
    predicted_results = predict(weights, structure, data)
    return np.average(np.power(predicted_results-results, 2))


def fit(structure, dataset):
    bias_slice_point = 0
    previous = structure[0]
    for layer in structure[1:]:
        bias_slice_point += previous*layer
        previous = layer
    n_of_parameters = bias_slice_point + sum(structure[1:])
    weights =  \
        scipy.optimize.differential_evolution(partial(evaluate,
                                                      structure=structure,
                                                      dataset=dataset),
                                              popsize=1,
                                              bounds=[(-10.0, 10.0)]*n_of_parameters,
                                              maxiter=3000,
                                              disp=True,
                                              tol=1e-5,
                                              mutation=(0.0, 1.99),
                                              recombination=0.5)
    return weights.x


def main():
    structure = (1, 6, 6, 1)
    dataset = [([x], [func(x)]) for x in np.linspace(-1.0, 1.0, 100)]
    weights = fit(structure, dataset)
    dataset = [([x], [func(x)]) for x in
               [random.uniform(-1.0, 1.0) for _ in range(500)]]
    data = np.array([a[0] for a in dataset])
    results = [a[1] for a in dataset]
    prediction_list = predict(weights, structure, data)
    for prediction, result in zip(prediction_list, results):
        print(prediction, result)
    for point in dataset:
        plt.plot(point[0], point[1], 'o', color='black')
    for predicted, point in zip(prediction_list, dataset):
        plt.plot(point[0], predicted, 'o', color='red')

    plt.savefig(f'{random.random}')


if __name__ == '__main__':
    main()
