import json
import os

import numpy as np

from node import Node


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

    # Save each desired attribute to its own array
    all_verts = np.empty_like(all_pops, object)
    all_edges = np.empty_like(all_pops, object)
    all_returned_values = np.empty_like(all_pops, object)
    all_prev_fits = np.empty_like(all_pops)
    for test in range(all_pops.shape[0]):
        for run in range(all_pops.shape[1]):
            for gen in range(all_pops.shape[2]):
                for indiv in range(all_pops.shape[3]):
                    node = all_pops[test,run,gen,indiv]
                    all_verts[test,run,gen,indiv], all_edges[test,run,gen,indiv] = node.to_lists()
                    all_returned_values[test,run,gen,indiv] = node.returned_value
                    all_prev_fits[test,run,gen,indiv] = node.prev_fit
    # Reformat since content is homologous
    all_returned_values = np.array(all_returned_values)

    path = 'saves/' + kwargs['name'] + '/'
    os.makedirs(path, exist_ok=True)
    # np.save(path + 'pops', all_pops)
    np.save(path + 'verts', all_verts)
    np.save(path + 'edges', all_edges)
    np.save(path + 'returned_value', all_returned_values)
    np.save(path + 'prev_fit', all_prev_fits)
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
    # all_pops = np.load(path + 'pops.npy', allow_pickle=True)
    all_verts = np.load(path + 'verts.npy', allow_pickle=True)
    all_edges = np.load(path + 'edges.npy', allow_pickle=True)
    all_returned_values = np.load(path + 'returned_value.npy', allow_pickle=True)
    all_prev_fits = np.load(path + 'prev_fit.npy', allow_pickle=True)
    all_fits = np.load(path + 'fits.npy')
    with open(path + 'kwargs.json', 'rb') as f:
        kwargs = string_to_func(json.load(f))

    # Save each desired attribute to its own array
    all_pops = np.empty_like(all_verts)
    for test in range(all_pops.shape[0]):
        for run in range(all_pops.shape[1]):
            for gen in range(all_pops.shape[2]):
                for indiv in range(all_pops.shape[3]):
                    node = Node.from_lists(all_verts[test, run, gen, indiv], all_edges[test, run, gen, indiv])
                    node.returned_value = all_returned_values[test, run, gen, indiv]
                    node.prev_fit = all_prev_fits[test, run, gen, indiv]
                    all_pops[test, run, gen, indiv] = node

    return all_pops, all_fits, kwargs


# if __name__ == '__main__':
#     all_pops, all_fits, kwargs = load_all('noop')
#     save_all(all_pops, all_fits, kwargs)