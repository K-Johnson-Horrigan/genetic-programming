import math

import networkx as nx
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from utils import load_all


# All functions relevant to saving, loading, and plotting.

def plot_min_fit(all_pops, all_fits, title=None, legend_title=None, **kwargs):
    fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    labels = [k[0] for k in kwargs['test_kwargs']]
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
    plt.legend(title=legend_title)
    plt.show()


def plot_nodes(nodes, fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot all given nodes and the fitness function"""
    xs = np.linspace(*kwargs['domains'][0])
    # Plot target function if given
    if fitness_func is not None:
        label = f'${str(kwargs['target_func'](sp.Symbol("x"))).replace("**","^")}$'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys)
    # Plot nodes
    for i,node in enumerate(nodes):
        if labels is None:
            # Label is the simplified expression if the label provided is None
            label = f'${str(node(sp.Symbol("x"))).replace("**","^")}$'
        else:
            label = f'\"{kwargs['test_kwargs'][i+1]}\", Fitness = {fitness_func([node], **kwargs)[0]:.3f}'
        node_ys = [node(i) for i in xs]
        plt.scatter(xs, node_ys, label=label)
        plt.plot(xs, node_ys)
    plt.title(title)
    plt.legend(title=legend_title)
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


def plot_tree(node, theta0=0.0, theta1=1.0, r=0, initial=True, verts=None, edges=None, pos=None, verts2=None):

    if initial:
        edges = [] if initial else edges
        verts = [node]
        verts2 = [[0]]
        pos = [(0,0)]
        # index = [index for index,n in enumerate(verts) if n is node][0]

    if node.temp_index is None: node.temp_index = len(verts) - 1

    theta = theta0 + 0.5 * (theta1 - theta0)

    # xx = math.cos(math.pi * 2 * theta) * r
    # yy = math.sin(math.pi * 2 * theta) * r

    if len(node) > 0:
        sub_r = r + 1
        for i, child in enumerate(node):
            # Check if the node already exists in the plot
            # sub = [index for index,n in enumerate(verts) if n is child]
            # if len(sub) > 0:

            # Child has already been iterated over
            if child.temp_index is not None:
                edges.append((node.temp_index, child.temp_index))

            # Child has not been iterated over
            else:
                child.temp_index = len(verts)
                verts.append(child)
                d = node.depth()
                if len(verts2) == d: verts2.append([])
                verts2[d].append(child.temp_index)
                pos.append(None)
                # Call recursively
                child_theta0 = theta0 + i / (len(node)) * (theta1 - theta0)
                child_theta1 = theta0 + (i + 1) / (len(node)) * (theta1 - theta0)
                sub_theta, sub_r = plot_tree(child, child_theta0, child_theta1, sub_r, False, verts, edges, pos, verts2)
                sub_x = math.cos(math.pi * 2 * sub_theta) * sub_r
                sub_y = math.sin(math.pi * 2 * sub_theta) * sub_r
                # Update position
                pos[child.temp_index] = (sub_x, sub_y)
                edges.append((node.temp_index, child.temp_index))

    if not initial:
        return theta, r
    else:
        node.reset_index()
        # Alternate layout
        # for r in range(len(verts2)):
        #     for i in range(len(verts2[r])):
        #         theta = i / len(verts2[r])
        #         x = math.cos(math.pi * 2 * theta) * r
        #         y = math.sin(math.pi * 2 * theta) * r
        #         pos[verts2[r][i]] = (x, y)

        # Create networkxs graph
        fig, ax = plt.subplots()
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(verts)))
        G.add_edges_from(edges)
        G.nodes(data=True)

        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.arf_layout(G)
        # pos = nx.planar_layout(G)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=range(len(verts)),
            node_color='tab:blue',
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels = {key: str(node.value) for key,node in enumerate(verts)},
            font_color="whitesmoke"
        )
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=10,
            # edge_color = range(G.number_of_edges()),
            # edge_cmap = plt.cm.gist_rainbow,
            width=2,
            alpha=0.5,
        )
        plt.show()


def plot_results(all_pops, all_fits, **kwargs):
    """Plot all standard plots"""
    plot_min_fit(all_pops, all_fits, title='', **kwargs)
    plot_best(all_pops, all_fits, title='Best Overall', **kwargs)
    # plot_size(all_pops, all_fits, title='Best Overall', **kwargs)


if __name__ == '__main__':
    name = 'logical_or'
    all_pops, all_fits, kwargs = load_all(name)
    plot_results(all_pops, all_fits, **kwargs)