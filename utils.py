import numpy as np
from collections import namedtuple


def orig_func(x):
    return np.power(x, 2) * np.sin(x) + 50 * np.sin(2 * x)


def func(x):
    return 0.005 * orig_func(10.0 * x) + 0.5


activation = {
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0) * 1),
    'sigmoid': (lambda x: np.divide(1, 1 + np.exp(-x)), lambda x: np.divide(np.exp(x), np.power(1 + np.exp(x), 2))),
    'gaussian': (lambda x: np.exp(-np.power(x, 2)), lambda x: np.multiply(-2 * x, np.exp(-np.power(x, 2)))),
    'linear': (lambda x: x, lambda x: 1)
}

Layer = namedtuple("Layer", "weights biases activ_func deriv_func")
