import os

import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

# All functions relevant to saving, loading, and plotting.

def plot_min(all_pops, all_fits, labels, title, label_title, **kwargs):
    """Display the history"""

    fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))

    # Largest and smallest values of all results and trials
    # true_max_y = np.min(f, axis=(0,2))
    # true_min_y = np.max(f, axis=(0,2))
    # ax.fill_between(x, true_min_y, true_max_y, alpha=.5, linewidth=0)

    for test in range(all_fits.shape[0]):
        # Plot smallest fitness value
        y = np.min(all_fits[test], axis=(0,2))
        plt.plot(x, y, label=labels[test])
        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)

    plt.title(title)
    ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Min Fitness Value')
    plt.legend(title=label_title)
    plt.show()



def plot_size(all_pops, all_fits, labels, title, label_title, **kwargs):
    """Display the history"""
    fig, ax = plt.subplots()
    all_fits = np.array(all_fits)
    x = np.array(range(all_fits.shape[2]))

    all_fits = np.vectorize(lambda x: x.depth())
    all_fits = all_fits()

    for test in range(all_fits.shape[0]):
        # Plot smallest fitness value
        # y = np.min(all_fits[test], axis=(0,2))
        # plt.plot(x, y, label=labels[test])
        # Scatter plot all points
        xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        yy = all_fits[test].ravel()
        plt.scatter(xx, yy, 0.1)

    plt.title(title)
    ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Min Fitness Value')
    plt.legend(title=label_title)
    plt.show()


def plot_nodes(nodes, fitness_func, title=None, label_title=None, labels=None, **kwargs):
    """Plot all given nodes"""
    x = np.linspace(*kwargs['x_linspace'])
    label = f'${str(kwargs['function'](sp.Symbol("x"))).replace("**","^")}$'
    plt.scatter(x, kwargs['function'](x), label=label)
    plt.plot(x, kwargs['function'](x))
    for i,node in enumerate(nodes):
        if labels is None:
            label = f'${str(node(sp.Symbol("x"))).replace("**","^")}$'
        else:
            label = f'\"{labels[i]}\", Fitness = {fitness_func([node], **kwargs)[0]:.3f}'
        plt.scatter(x, [node(i, algebraic=False) for i in x], label=label)
        plt.plot(x, [node(i, algebraic=False) for i in x])
    plt.title(title)
    plt.legend(title=label_title)
    plt.show()


def plot_best(all_pops, all_fits, run=None, gen=slice(None), **kwargs):
    """Plot the best result of the given run and gen"""
    if run is None:
        runs = range(all_pops.shape[0])
    elif type(run) is not list:
        runs = [run]
    else:
        runs = run
    nodes = []

    # Iterate over all runs
    for run in runs:
        i = all_fits[run,slice(None),gen,:].argmin()
        node = all_pops[run,slice(None),gen,:].flatten()[i]
        print(node)
        # print(node.simplify())
        fit = all_fits[run,slice(None),gen,:].flatten()[i]
        nodes.append(node)
        # print(np.unravel_index(i, all_fits[run,gen,:].shape))
    plot_nodes(nodes, **kwargs)


def plot_sims(all_pops, all_fits, **kwargs):
    """Create all plots"""
    # Display the current histories
    plot_min(all_pops, all_fits, title='', **kwargs)
    plot_best(all_pops, all_fits, title='Best Overall', **kwargs)
    # plot_size(all_pops, all_fits, title='Best Overall', **kwargs)


def save_all(all_pops, all_fits, kwargs):
    path = 'saves/' + kwargs['name'] + '/'
    os.makedirs(path, exist_ok=True)
    np.save(path + 'pops', all_pops)
    np.save(path + 'fits', all_fits)
    np.save(path + 'kwargs', np.array([kwargs]))


def load_all(name):
    path = 'saves/' + name + '/'
    all_pops = np.load(path + 'pops.npy', allow_pickle=True)
    all_fits = np.load(path + 'fits.npy')
    # kwargs = np.load(path + 'kwargs.npy', allow_pickle=True)[0]
    kwargs = None
    return all_pops, all_fits, kwargs


if __name__ == '__main__':
    name = 'mod2'
    import gp
    all_pops, all_fits, kwargs = load_all(name)
    kwargs = gp.kwargs
    print('Loaded Data')
    plot_sims(all_pops, all_fits, **kwargs)