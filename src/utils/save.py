"""Functions used to save and load data."""

import glob
import json
import os

import numpy as np

from src.utils.utils import to_tuple


def save_kwargs(**kwargs):
    FUNC_PREFIX = 'src.genetics'
    def func_to_string(obj):
        """Recursively replace functions with its name preceded by src.genetics"""
        if type(obj) == dict:
            obj = obj.copy()
            for key in obj:
                obj[key] = func_to_string(obj[key])
            return obj
        elif type(obj) == list:
            obj = obj.copy()
            for i, item in enumerate(obj):
                obj[i] = func_to_string(item)
            return obj
        elif type(obj) == np.ndarray:
            return to_tuple(obj)
        elif hasattr(obj, '__name__'):
            return f'{FUNC_PREFIX}.{obj.__name__}'
        else:
            return obj
    kwargs_path = f'{kwargs['saves_path']}{kwargs['name']}/'
    os.makedirs(kwargs_path, exist_ok=True)
    print(f'Saving kwargs to {kwargs_path}kwargs.json')
    with open(kwargs_path + 'kwargs.json', 'w') as f:
        json.dump(func_to_string(kwargs.copy()), f, indent=4)


def save_run(test_path, pops, fits, **kwargs):
    # Each test is saved in its own directory which is passed through the path
    os.makedirs(test_path, exist_ok=True)
    test_path = f'{test_path}/{kwargs["seed"]}/'
    print(f'Saving run to {test_path}')
    os.makedirs(test_path, exist_ok=True)
    np.save(test_path + 'pops', pops)
    np.save(test_path + 'fits', fits)


def load_kwargs(name, saves_path):
    FUNC_PREFIX = 'src.genetics'
    def string_to_func(obj):
        """Recursively replace strings preceded by src.genetics with the imported function of the same name"""
        if type(obj) is dict:
            for key in obj:
                obj[key] = string_to_func(obj[key])
        elif type(obj) is list:
            for i, item in enumerate(obj):
                obj[i] = string_to_func(item)
        elif type(obj) is type('') and obj.startswith(FUNC_PREFIX + '.'):
            # module = __import__(GP_FILE)
            import src.models as module
            return getattr(module, obj[len(FUNC_PREFIX) + 1:])
        return obj
    path = f'{saves_path}{name}/kwargs.json'
    print(f'Loading kwargs from {path}')
    with open(path, 'rb') as f:
        kwargs = string_to_func(json.load(f))
    kwargs['saves_path'] = saves_path
    return kwargs


def load_runs(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    pops = []
    fits = []
    tests = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test in tests:
        pops.append([])
        fits.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test}/*/'
        for run_file_name in glob.glob(test_path):
            print(f'Loading run from {run_file_name}')
            pops[-1].append(np.load(run_file_name+'pops.npy', allow_pickle=True))
            fits[-1].append(np.load(run_file_name+'fits.npy'))
    # pops = np.array(pops, dtype=[('verts','object'),('edges','object')])
    pops = np.array(pops, dtype=object)
    fits = np.array(fits)
    return pops, fits


def load_fits(**kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    fits = []
    test_names = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test_name in test_names:
        fits.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
        for run_file_name in sorted(glob.glob(test_path)):
            print(f'Loading fitness from {run_file_name}')
            fits[-1].append(np.load(run_file_name+'fits.npy'))
    fits = np.array(fits)
    return fits


def load_pops(test, run, **kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    pops = []
    test_names = [test[0] for test in kwargs['test_kwargs'][1:]]
    for test_name in test_names:
        pops.append([])
        test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
        for run_file_name in glob.glob(test_path):
            print(f'Loading run from {run_file_name}')
            pops[-1].append(np.load(run_file_name+'pops.npy', allow_pickle=True))
    # pops = np.array(pops, dtype=[('verts','object'),('edges','object')])
    pops = np.array(pops, dtype=object)
    fits = np.array(fits)
    return pops, fits


def load_pop(test, run, **kwargs):
    """Returns a 4D array of all individuals and fitness values"""
    test_name = kwargs['test_kwargs'][1:][test][0]
    test_path = f'{kwargs['saves_path']}{kwargs['name']}/data/{test_name}/*/'
    run_file_name = sorted(glob.glob(test_path))[run]
    print(f'Loading population from {run_file_name}')
    pop = np.load(run_file_name+'pops.npy', allow_pickle=True)
    return pop



if __name__ == '__main__':
    # name = 'unstable_self_rep_0'
    name = 'mult_1'
    kwargs = load_kwargs(name, '../../saves/')
    fits = load_fits(**kwargs)

    pop = load_pop(0, 0, **kwargs)



    pass

