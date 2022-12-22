experiments = [
    {
        'name': '1,2,2,1_gradient',
        'path': '1,2,2,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,2,2,1_evolution',
        'path': '1,2,2,1_evolution_test',
        'method': 'evolution',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 3000,

    },
    {
        'name': '1,5,5,1_gradient',
        'path': '1,5,5,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 5, 5, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,5,5,1_evolution',
        'path': '1,5,5,1_evolution_test',
        'method': 'evolution',
        'topology': (1, 5, 5, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 3000,

    },
    {
        'name': '1,10,10,1_gradient',
        'path': '1,10,10,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 2, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,10,10,1__evolution',
        'path': '1,10,10,1__evolution_test',
        'method': 'evolution',
        'topology': (1, 10, 10, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 1000,

    },
    {
        'name': '1,2,10,1_gradient',
        'path': '1,2,10,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 2, 10, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,2,10,1__evolution',
        'path': '1,2,10,1__evolution_test',
        'method': 'evolution',
        'topology': (1, 2, 10, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 1500,

    },
    {
        'name': '1,10,2,1_gradient',
        'path': '1,10,2,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 10, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,10,2,1_evolution',
        'path': '1,10,2,1_evolution_test',
        'method': 'evolution',
        'topology': (1, 10, 2, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 1500,

    },
    {
        'name': '1,20,20,1_gradient',
        'path': '1,20,20,1_gradient_test',
        'method': 'gradient',
        'topology': (1, 20, 20, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 10,
        'batch_size': 20,
        'learning_rate': 0.5,
    },
    {
        'name': '1,20,20,1_evolution',
        'path': '1,20,20,1_evolution_test',
        'method': 'evolution',
        'topology': (1, 20, 20, 1),
        'activ_func': ('gaussian', 'gaussian', 'linear'),
        'epoch': 1000,

    },
]
