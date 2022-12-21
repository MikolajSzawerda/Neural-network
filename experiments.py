experiments = [
    {
        'name': 'Dupa',
        'path': 'test',
        'method': 'gradient',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': 'Dupa2',
        'path': 'test2',
        'method': 'evolution',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
]