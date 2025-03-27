import json
import os

import numpy as np


def save_all(all_pops, all_fits, kwargs):
    path = 'saves/' + kwargs['name'] + '/'
    os.makedirs(path, exist_ok=True)
    np.save(path + 'pops', all_pops)
    np.save(path + 'fits', all_fits)
    # Copy kwargs so that functions can be replaced
    kwargs = kwargs.copy()
    # Replace functions with names
    for kwarg in kwargs:
        if kwarg.endswith('_func'):
            kwargs[kwarg] = kwargs[kwarg].__name__
    # Replace test kwarg functions with names
    for i,test_kwarg in enumerate(kwargs['test_kwargs'][0]):
        if test_kwarg.endswith('_func'):
            for j in range(1, len(kwargs['test_kwargs'])):
                kwargs['test_kwargs'][j][i] = kwargs['test_kwargs'][j][i].__name__
    # Save as JSON
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
    # Replace names with functions
    for kwarg in kwargs:
        if kwarg.endswith('_func'):
            kwargs[kwarg] = getattr(gp, kwargs[kwarg])
    # Replace test kwarg names with functions
    for i, test_kwarg in enumerate(kwargs['test_kwargs'][0]):
        if test_kwarg.endswith('_func'):
            for j in range(1, len(kwargs['test_kwargs'])):
                kwargs['test_kwargs'][j][i] = getattr(gp, kwargs['test_kwargs'][j][i])
    return all_pops, all_fits, kwargs