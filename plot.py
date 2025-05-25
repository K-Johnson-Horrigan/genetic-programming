import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from node import Node
from utils import load_kwargs, load_runs

""" All functions relevant to saving, loading, and plotting."""


def plot_nodes(nodes, result_fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot all given nodes and the fitness function"""

    # Only plot the first domain
    xs = np.linspace(*kwargs['domains'][0])

    # Plot target function if given
    if 'target_func' in kwargs and result_fitness_func is not None:
        label = 'Target Function'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys, lw=5)

    # Plot nodes
    for i, node in enumerate(nodes):
        # Determine label based on what info is known
        if labels is not None:
            label = labels[i]
        elif 'test_kwargs' in kwargs:
            label = kwargs['test_kwargs'][i + 1][0]
        else:
            label = ''
        if 'target_func' in kwargs and result_fitness_func is not None:
            label += f' Fitness = {result_fitness_func([node], **kwargs)[0]:f}'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    plt.title(title)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()


def plot_min_fit(all_pops, all_fits, title=None, legend_title=None, **kwargs):
    fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    for test in range(all_fits.shape[0]):
        y = np.mean(np.min(all_fits[test], axis=2), axis=0)
        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)
    plt.title(title)
    # ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Average Min Fitness Value')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/Fits.png')
    plt.show()


def plot_means(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(values[test], axis=(0,2))
        xs = np.array(range(values.shape[2]))
        plt.plot(xs, ys, label=label)
        # ys_std = ys.std()
        # ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_medians(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        xs = np.array(range(values.shape[2]))
        ys = np.median(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        ys = np.mean(values[test], axis=(0,2))
        plt.plot(xs, ys, label=label)

        q1 = np.quantile(values[test], 0.25, axis=(0,2))
        q3 = np.quantile(values[test], 0.75, axis=(0,2))
        ax.fill_between(xs, q1, q3, alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_hist(values, ylabel):
    fig, ax = plt.subplots()
    for test in range(values.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        xs = values[test, :, -1].ravel()
        # ax.boxplot(xs,
        #     positions=[test],
        #     label=label,
        #     patch_artist=True,
        #     # showmeans=False,
        #     # showfliers=False,
        #     # medianprops={"color": "white", "linewidth": 0.5},
        #     # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
        #     # whiskerprops={"color": "C0", "linewidth": 1.5},
        #     # capprops={"color": "C0", "linewidth": 1.5}
        # )

        label = kwargs['test_kwargs'][test + 1][0]
        xs = values[test, :, -1].ravel()
        ax.boxplot(xs,
                   positions=[test],
                   label=label,
                   patch_artist=True,
                   # showmeans=False,
                   # showfliers=False,
                   # medianprops={"color": "white", "linewidth": 0.5},
                   # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                   # whiskerprops={"color": "C0", "linewidth": 1.5},
                   # capprops={"color": "C0", "linewidth": 1.5}
                   )
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'saves/{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


# def plot_size(all_pops, all_fits, **kwargs):
#     graph_size = np.vectorize(lambda x: len(x[0]))(all_pops)
#     for test in range(all_fits.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         ys = np.mean(graph_size[test], axis=(0,2))
#         xs = np.array(range(all_fits.shape[2]))
#         plt.plot(xs, ys, label=label)
#     plt.xlabel('Generation')
#     plt.ylabel('Average Number of Nodes')
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'saves/{kwargs["name"]}/plots/Size.png')
#     plt.show()


# def plot_quality_gain(all_pops, all_fits, **kwargs):
#     """Plot the expected change in fitness from parent to offspring"""
#     # Calculate quality gain and replace invalid values with zero as no meaningful change occurred
#     quality_gain = all_fits - np.vectorize(lambda x: x.prev_fit, otypes=[float])(all_pops)
#     quality_gain[abs(quality_gain) > 1e100] = 0
#     quality_gain = np.nan_to_num(quality_gain, nan=0)
#     for test in range(all_fits.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         ys = np.mean(quality_gain[test], axis=(0,2))
#         xs = np.array(range(all_fits.shape[2]))
#         plt.plot(xs, ys, label=label)
#     # plt.gca().set_yscale('log')
#     plt.xlabel('Generation')
#     plt.ylabel('Quality Gain')
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'saves/{kwargs["name"]}/plots/Quality Gain.png')
#     plt.show()
#
#
# def plot_success_rate(all_pops, all_fits, legend_title=None, **kwargs):
#     """Plot the probability that an offspring is better than their parent"""
#     # Calculate success rate and replace invalid values with zero as no meaningful change occurred
#     success_rate = all_fits < np.vectorize(lambda x: x.prev_fit, otypes=[float])(all_pops)
#     success_rate[abs(success_rate) > 1e100] = 0
#     success_rate = np.nan_to_num(success_rate, nan=0)
#     for test in range(all_fits.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         ys = np.mean(success_rate[test], axis=(0,2))
#         xs = np.array(range(all_fits.shape[2]))
#         plt.plot(xs, ys, label=label)
#     plt.xlabel('Generation')
#     plt.ylabel('Success Rate')
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'saves/{kwargs["name"]}/plots/Success Rate.png')
#     plt.show()
#
#
# def plot_effective(all_pops, all_fits, legend_title=None, **kwargs):
#     """Plot the percentage of operations that have a non-zero semantic vector"""
#     effective_code = np.vectorize(lambda x: x.effective_code())(all_pops)
#     for test in range(all_fits.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         ys = np.mean(effective_code[test], axis=(0,2))
#         xs = np.array(range(all_fits.shape[2]))
#         plt.plot(xs, ys, label=label)
#     plt.xlabel('Generation')
#     plt.ylabel('% Active Traversed Nodes')
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'saves/{kwargs["name"]}/plots/Active Nodes.png')
#     plt.show()
#
#
# def plot_noop_size(all_pops, all_fits, legend_title=None, **kwargs):
#     """Plot the percentage of operations that have a non-zero semantic vector"""
#     noop_size = np.vectorize(
#         lambda x:
#             sum([len(x[i].nodes()) for i in range(1,len(x))]) / len(x.nodes()) if x.value == 'noop' else 0
#     )(all_pops)
#     for test in range(all_fits.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         ys = np.mean(noop_size[test], axis=(0,2))
#         xs = np.array(range(all_fits.shape[2]))
#         plt.plot(xs, ys, label=label)
#     plt.xlabel('Generation')
#     plt.ylabel('% No-Op Traversed Nodes')
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'saves/{kwargs["name"]}/plots/Noop Nodes.png')
#     plt.show()


#
# Tables
#

def table_best(all_pops, all_fits, **kwargs):
    """Plot the best result of the given run and gen"""
    xs = [np.linspace(*domain) for domain in kwargs['domains']]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_true = np.array([[kwargs['target_func'](*list(x))] for x in xs])
    table = np.concat((xs, y_true), axis=1)
    # Iterate over all runs
    for run in range(len(kwargs['test_kwargs']) - 1):
        i = all_fits[run, :, :, :].argmin()
        node = all_pops[run, :, :, :].flatten()[i]
        y_node = [[node(*x)] for x in xs]
        tab = np.concat((table, y_node), axis=1)
        print('\n', node, sep='')
        for row in tab:
            print(('f(' + ', '.join(['{}'] * len(kwargs['domains'])) + ') = {} | {}').format(*row))


#
# Graphs
#

def plot_graph(node, layout='topo', scale=1, title=None, **kwargs):
    """Plot the node as a graph"""

    def to_graph(node, verts=None, edges=None, vert_props=None, edge_props=None):
        """Identical to Node.to_lists() but returns extra values"""
        if verts is None:
            node.reset_index()
            verts, edges = [], []
            vert_props, edge_props = [], []
        if node.temp_index is None:
            node.temp_index = len(verts)
            verts.append(node.value)
            # vert_props.append()
            for i, child in enumerate(node.children):
                to_graph(child, verts, edges, vert_props, edge_props)
                edges.append((node.temp_index, child.temp_index))
                edge_props.append(i)
        return verts, edges, vert_props, edge_props

    verts, edges, vert_props, edge_props = to_graph(node)
    # Create networkxs graph
    fig, ax = plt.subplots()
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    connectionstyle = [f"arc3,rad={r}" for r in [0, .5]]
    if layout == 'traversal':
        # Traversal layout
        pos = sorted([(n.temp_index, n.depth()) for n in node.nodes()])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel('Traversal Order')
        plt.ylabel('Depth')
    else:
        # Topological layout
        for layer, ns in enumerate(nx.topological_generations(G)):
            for n in ns:
                G.nodes[n]["layer"] = layer
        pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=range(len(verts)),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle=[["-|>", "->"][i] for i in edge_props],
        edgelist=edges, # Specify edge order
        connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     connectionstyle=connectionstyle,
    #     edge_labels = {edges[key]: label for key,label in enumerate(edge_props)},
    #     alpha=0.5,
    #     label_pos=0.0,
    #     node_size=24000 * scale,
    #     bbox=None,
    # )
    plt.suptitle(f'${node.latex()}$')
    plt.title(title)
    if 'result_fitness_func' in kwargs:
        plt.legend(title=f'Fitness = {kwargs['result_fitness_func']([node], **kwargs)[0]}')
    if 'name' in kwargs:
        plt.savefig(f'saves/{kwargs["name"]}/plots/{title}.png')
    plt.show()


#
# Control
#

def get_best(all_pops, all_fits, gen=-1, **kwargs):
    """Get the best result of the given run and gen"""
    nodes = []
    # Iterate over all runs
    for run in range(all_pops.shape[0]):
        i = all_fits[run, slice(None), gen, :].argmin()
        node = all_pops[run, slice(None), gen, :].flatten()[i]
        nodes.append(Node.from_lists(*node))
    return nodes


def plot_results(all_pops, all_fits, **kwargs):
    """Plot all standard plots"""
    path = f'saves/{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')

    # plot_min_fit(all_pops, all_fits, title='', **kwargs)

    # Plot best
    best = get_best(all_pops, all_fits, **kwargs)
    # if len(kwargs['domains']) == 1:
    #     plot_nodes(best, **kwargs)
    # else:
    #     table_best(all_pops, all_fits, title='Best Overall', **kwargs)

    # plot_means(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_medians(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_hist(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')

    # plot_size(all_pops, all_fits, **kwargs)
    # plot_quality_gain(all_pops, all_fits, **kwargs)
    # plot_success_rate(all_pops, all_fits, **kwargs)
    # plot_effective(all_pops, all_fits, **kwargs)
    # plot_noop_size(all_pops, all_fits, **kwargs)

    for i, node in enumerate(best):
        print(node)
        title = 'Best Graph (' + kwargs['test_kwargs'][i + 1][0] + ')'
        plot_graph(node, title=title, **kwargs)


if __name__ == '__main__':
    kwargs = load_kwargs('debug')
    pops, fits = load_runs(**kwargs)
    plot_results(pops, fits, **kwargs)

    # x = Node('x')
    # f = x / x + x * x
    # f = f.to_tree()
    # plot_graph(f)



