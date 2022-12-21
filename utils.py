import numpy as np
from collections import namedtuple

activation = {
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0) * 1),
    'sigmoid': (lambda x: np.divide(1, 1 + np.exp(-x)), lambda x: np.divide(np.exp(x), np.power(1 + np.exp(x), 2))),
    'gaussian': (lambda x: np.exp(-np.power(x, 2)), lambda x: np.multiply(-2 * x, np.exp(-np.power(x, 2)))),
    'linear': (lambda x: x, lambda x: 1)
}

Layer = namedtuple("Layer", "weights biases activ_func deriv_func")
