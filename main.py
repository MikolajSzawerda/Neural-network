import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import pandas as pd

def orig_func(x):
    return np.power(x, 2) * np.sin(x) + 50 * np.sin(2 * x)


def func(x):
    return 0.005 * orig_func(10.0 * x) + 0.5


if __name__ == '__main__':
    params = {
        'method': 'evolution',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 100,
        'batch_size': 20,
        'learning_rate': 0.5,
    }
    x = np.linspace(-1, 1, 100)
    np.random.shuffle(x)
    y = func(x)
    split_index = int(0.7 * x.shape[0])
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    nn = NeuralNetwork(**params)
    nn.train(x, y)
    mse = nn.get_mse_progress(x_test, y_test)
    pd.DataFrame(mse).to_csv("results/test_mse.csv")
    aprox = pd.DataFrame(data=[
        x_test,
        np.asarray(nn.predict(np.asmatrix(x_test))).reshape(-1),
        func(x_test)
    ]
    ).transpose().sort_values(by=0)
    aprox.rename(columns={0: 'x', 1: 'y_predict', 2: 'y'}, inplace=True)
    aprox.to_csv("results/test_predict.csv")
