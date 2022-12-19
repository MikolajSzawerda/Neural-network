import math
import random
import scipy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+math.exp(-x))


def ReLU(x):
    return (max(0, x))


def gaussian(x):
    return math.exp(-(x**2))


def f(x):
    return ((x**2))*math.sin(x) + 100*math.sin(x)*math.cos(x)


def small_network_opt(weights, dataset, structure):
    loss = 0
    for input, result in dataset:
        predicted_result = small_network(input, weights, structure)
        loss += (result - predicted_result[0])**2
    return loss/len(dataset)


# def small_network(input, weights):

#     n1 = input*weights[0]+weights[1]
#     n2 = input*weights[2]+weights[3]
#     n3 = input*weights[4]+weights[5]

#     n1 = ReLU(n1)
#     n2 = ReLU(n2)
#     n3 = ReLU(n3)

#     n4 = n1*weights[6] + n2*weights[9] + n3*weights[12] + weights[15]
#     n5 = n1*weights[7] + n2*weights[10] + n3*weights[13] + weights[16]
#     n6 = n1*weights[8] + n2*weights[11] + n3*weights[14] + weights[17]

#     n4 = ReLU(n4)
#     n5 = ReLU(n5)
#     n6 = ReLU(n6)

#     predicted_result = n4*weights[18]+ n5*weights[19] + n6*weights[20] + weights[21]
#     return predicted_result


def network_layer(input, weights, biases, is_with_activation):
    neurons = list()
    for neuron_weights, bias in zip(weights, biases):
        neuron_value = 0
        for value, weight in zip(input, neuron_weights):
            neuron_value += value*weight
        neuron_value += bias
        if is_with_activation:
            neuron_value = ReLU(neuron_value)
        neurons.append(neuron_value)
    return neurons


def slice_weights(weights, structure):
    sliced_weights = list()
    biases = list()
    slice_point = 0
    prev = structure[0]
    for n_of_neurons in structure[1:]:
        layer_weights = list()
        for _ in range(n_of_neurons):
            layer_weights.append(weights[slice_point:slice_point+prev])
            slice_point += prev
        sliced_weights.append(layer_weights)
        prev = n_of_neurons

    for n_of_neurons in structure[1:]:
        biases.append(weights[slice_point:slice_point+n_of_neurons])
        slice_point += n_of_neurons

    return sliced_weights, biases


def small_network(input, weights, structure):

    network_weights, biases = slice_weights(weights, structure)

    layer_result = network_layer(input, network_weights[0], biases[0], True)

    for layer_weights, layer_biases in zip(network_weights[1:-1], biases[1:-1]):
        layer_result = network_layer(layer_result, layer_weights, layer_biases, True)

    return network_layer(layer_result, network_weights[-1], biases[-1], False)


def main():
    structure = [1, 4, 4, 1]
    dataset = [([x], f(x)) for x in [random.uniform(-3.0, 3.0) for _ in range(100)]]
    weights = scipy.optimize.differential_evolution(partial(small_network_opt, dataset=dataset, structure=structure), bounds=[(-10.0, 10.0)]*33, maxiter=100)
    print(weights)
    dataset = [([x], f(x)) for x in [random.uniform(-3.0, 3.0) for _ in range(100)]]
    prediction_list = list()
    for input, result in dataset:
        prediction = small_network(input, weights.x, structure)
        print(prediction, result)
        prediction_list.append(prediction[0])
    for point in dataset:
        plt.plot(point[0], point[1], 'o', color='black')
    for predicted, point in zip(prediction_list, dataset):
        plt.plot(point[0], predicted, 'o', color='red')

    plt.show()




if __name__ == '__main__':
    main()
