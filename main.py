import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import pandas as pd
from experiments import experiments
import time
from utils import func


def path(folder, base, exp):
    return f'{folder}/{base}_{exp}.csv'


if __name__ == '__main__':
    params = {
        'method': 'gradient',
        'topology': (1, 20, 20, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 1000,
        'batch_size': 20,
        'learning_rate': 0.5,
    }
    x = np.linspace(-1, 1, 1000)
    np.random.shuffle(x)
    y = func(x)
    split_index = int(0.7 * x.shape[0])
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    experiments_data = []
    for experiment in experiments:
        nn = NeuralNetwork(**experiment)
        start = time.process_time()
        nn.train(x, y)
        end = time.process_time()
        mse = nn.get_mse_progress(x_test, y_test)
        pd.DataFrame({'it': list(range(len(mse))),
                      'mse': mse}).to_csv(path('results', experiment['path'], 'mse'))
        aprox = pd.DataFrame(data=[
            x_test,
            np.asarray(nn.predict(np.asmatrix(x_test))).reshape(-1),
            func(x_test)
        ]
        ).transpose().sort_values(by=0)
        aprox.rename(columns={0: 'x', 1: 'y_predict', 2: 'y'}, inplace=True)
        aprox.to_csv(path('results', experiment['path'], 'predict'))
        experiments_data.append((experiment['name'],
                                 end-start,
                                 mse[-1],
                                 experiment['path'],
                                 path('results', experiment['path'], 'mse'),
                                 path('results', experiment['path'], 'predict')))
    pd.DataFrame(experiments_data, columns=['name', 'time', 'mse', 'path', 'mse_path', 'predict_path']).to_csv("results/experiments.csv")


