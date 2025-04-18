import json
import os

import numpy as np


def save_all(all_pops, all_fits, kwargs):

    def func_to_string(obj):
        """Recursively replace strings preceded by $ to a function of the same name from gp"""
        if type(obj) == dict:
            for key in obj:
                obj[key] = func_to_string(obj[key])
        elif type(obj) == list:
            for i, item in enumerate(obj):
                obj[i] = func_to_string(item)
        elif hasattr(obj, '__name__'):
            return '$' + obj.__name__
        return obj

    path = 'saves/' + kwargs['name'] + '/'
    os.makedirs(path, exist_ok=True)
    np.save(path + 'pops', all_pops)
    np.save(path + 'fits', all_fits)
    with open(path + 'kwargs.json', 'w') as f:
        json.dump(func_to_string(kwargs.copy()), f, indent=4)



def load_all(name):
    import gp

    def string_to_func(obj):
        """Recursively replace strings preceded by $ to a function of the same name from gp"""
        if type(obj) is dict:
            for key in obj:
                obj[key] = string_to_func(obj[key])
        elif type(obj) is list:
            for i, item in enumerate(obj):
                obj[i] = string_to_func(item)
        elif type(obj) is type('') and obj.startswith('$'):
            return getattr(gp, obj[1:])
        return obj

    path = 'saves/' + name + '/'
    all_pops = np.load(path + 'pops.npy', allow_pickle=True)
    all_fits = np.load(path + 'fits.npy')
    with open(path + 'kwargs.json', 'rb') as f:
        kwargs = string_to_func(json.load(f))
    return all_pops, all_fits, kwargs