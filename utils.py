import json
import os

import numpy as np


def save_all(all_pops, all_fits, kwargs):
    path = 'saves/' + kwargs['name'] + '/'
    os.makedirs(path, exist_ok=True)
    np.save(path + 'pops', all_pops)
    np.save(path + 'fits', all_fits)
    kwargs = kwargs.copy()
    for kwarg in kwargs:
        if kwarg.endswith('_func'):
            kwargs[kwarg] = kwargs[kwarg].__name__
    with open(path + 'kwargs.json', 'w') as f:
        json.dump(kwargs, f, indent=4)


def load_all(name):
    import gp
    path = 'saves/' + name + '/'
    all_pops = np.load(path + 'pops.npy', allow_pickle=True)
    all_fits = np.load(path + 'fits.npy')
    # kwargs = np.load(path + 'kwargs.npy', allow_pickle=True)[0]
    with open(path + 'kwargs.json', 'rb') as f:
        kwargs = json.load(f)
    for kwarg in kwargs:
        if kwarg.endswith('_func'):
            kwargs[kwarg] = getattr(gp, kwargs[kwarg])
    return all_pops, all_fits, kwargs