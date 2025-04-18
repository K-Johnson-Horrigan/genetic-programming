import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from utils import load_all


# All functions relevant to saving, loading, and plotting.


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
    for i,node in enumerate(nodes):
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
    plt.show()


def plot_min_fit(all_pops, all_fits, title=None, legend_title=None, **kwargs):
    fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    # labels = [k[0] for k in kwargs['test_kwargs'][1:]]
    # Largest and smallest values of all results and trials
    # true_max_y = np.min(f, axis=(0,2))
    # true_min_y = np.max(f, axis=(0,2))
    # ax.fill_between(x, true_min_y, true_max_y, alpha=.5, linewidth=0)

    for test in range(all_fits.shape[0]):
        # Plot smallest fitness value
        y = np.min(all_fits[test], axis=(0,2))
        plt.plot(x, y, label=kwargs['test_kwargs'][test+1][0])
        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)

    plt.title(title)
    # ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Min Fitness Value')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


def plot_size(all_pops, all_fits, legend_title=None, **kwargs):
    non_effective_code = np.vectorize(lambda x: len(x.nodes()))(all_pops)
    # non_effective_code[abs(non_effective_code) > 1e100] = 0
    for test in range(all_fits.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(non_effective_code[test], axis=(0,2))
        xs = np.array(range(all_fits.shape[2]))
        plt.plot(xs, ys, label=label)
    # ax.set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Number of Nodes')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


def plot_quality_gain(all_pops, all_fits, legend_title=None, **kwargs):
    """Plot the expected change in fitness from parent to offspring"""
    # Calculate quality gain and replace invalid values with zero as no meaningful change occurred
    quality_gain = all_fits - np.vectorize(lambda x: x.prev_fit, otypes=[float])(all_pops)
    quality_gain[abs(quality_gain) > 1e100] = 0
    quality_gain = np.nan_to_num(quality_gain, nan=0)
    for test in range(all_fits.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(quality_gain[test], axis=(0,2))
        xs = np.array(range(all_fits.shape[2]))
        plt.plot(xs, ys, label=label)
    # plt.gca().set_yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Quality Gain')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


def plot_success_rate(all_pops, all_fits, legend_title=None, **kwargs):
    """Plot the probability that an offspring is better than their parent"""
    # Calculate success rate and replace invalid values with zero as no meaningful change occurred
    success_rate = all_fits < np.vectorize(lambda x: x.prev_fit, otypes=[float])(all_pops)
    success_rate[abs(success_rate) > 1e100] = 0
    success_rate = np.nan_to_num(success_rate, nan=0)
    for test in range(all_fits.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(success_rate[test], axis=(0,2))
        xs = np.array(range(all_fits.shape[2]))
        plt.plot(xs, ys, label=label)
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


def plot_effective(all_pops, all_fits, legend_title=None, **kwargs):
    """Plot the percentage of operations that have a non-zero semantic vector"""
    effective_code = np.vectorize(lambda x: x.effective_code())(all_pops)
    for test in range(all_fits.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(effective_code[test], axis=(0,2))
        xs = np.array(range(all_fits.shape[2]))
        plt.plot(xs, ys, label=label)
    plt.xlabel('Generation')
    plt.ylabel('% Active Nodes')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


def plot_noop_size(all_pops, all_fits, legend_title=None, **kwargs):
    """Plot the percentage of operations that have a non-zero semantic vector"""
    noop_size = np.vectorize(
        lambda x:
            sum([len(x[i].nodes()) for i in range(1,len(x))]) / len(x.nodes()) if x.value == 'noop' else 0
    )(all_pops)
    for test in range(all_fits.shape[0]):
        label = kwargs['test_kwargs'][test + 1][0]
        ys = np.mean(noop_size[test], axis=(0,2))
        xs = np.array(range(all_fits.shape[2]))
        plt.plot(xs, ys, label=label)
    plt.xlabel('Generation')
    plt.ylabel('% No-Op Traversed Nodes')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.show()


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
        i = all_fits[run,:,:,:].argmin()
        node = all_pops[run,:,:,:].flatten()[i]
        y_node = [[node(*x)] for x in xs]
        tab = np.concat((table, y_node), axis=1)
        print('\n',node,sep='')
        for row in tab:
            print(('f(' + ', '.join(['{}']*len(kwargs['domains'])) + ') = {} | {}').format(*row))

#
# Graphs
#

# def plot_graph(node, layout=0, theta0=-.5, theta1=.5, r=0, initial=True, title=None, suptitle=None, verts=None, edges=None, pos=None, verts2=None):
#     """Plot the node as a tree"""
#
#     if layout==0:
#         simple_plot_graph(node, title=title, suptitle=suptitle)
#         return
#
#     if initial:
#         node.reset_index()
#         edges = [] if initial else edges
#         verts = [node]
#         verts2 = [[0]]
#         pos = [(0,0)]
#         # index = [index for index,n in enumerate(verts) if n is node][0]
#
#     if node.temp_index is None: node.temp_index = len(verts) - 1
#
#     theta = theta0 + 0.5 * (theta1 - theta0)
#
#     # xx = math.cos(math.pi * 2 * theta) * r
#     # yy = math.sin(math.pi * 2 * theta) * r
#
#     if len(node) > 0:
#         child_r = r + 1
#         for i, child in enumerate(node):
#             # Check if the node already exists in the plot
#             # sub = [index for index,n in enumerate(verts) if n is child]
#             # if len(sub) > 0:
#
#             # Child has already been iterated over
#             if child.temp_index is not None:
#                 edges.append((node.temp_index, child.temp_index))
#
#             # Child has not been iterated over
#             else:
#                 child.temp_index = len(verts)
#                 verts.append(child)
#                 pos.append(None)
#                 if layout == 2 or layout == 4:
#                     depth = node.depth()
#                     while len(verts2) <= depth: verts2.append([])
#                     verts2[depth].append(child.temp_index)
#                 # Call recursively
#                 child_theta0 = theta0 + i / (len(node)) * (theta1 - theta0)
#                 child_theta1 = theta0 + (i + 1) / (len(node)) * (theta1 - theta0)
#                 child_theta, child_r = plot_graph(child, layout, child_theta0, child_theta1, child_r, False, verts, edges, pos, verts2)
#                 if layout == 0:
#                     child_x = child_theta
#                     child_y = child_r
#                 else:
#                     child_x = math.cos(math.pi * 2 * child_theta) * child_r
#                     child_y = math.sin(math.pi * 2 * child_theta) * child_r
#
#                 # Update position
#                 pos[child.temp_index] = (child_x, child_y)
#                 edges.append((node.temp_index, child.temp_index))
#
#     if not initial:
#         return theta, r
#     else:
#         # Alternate layout
#         if layout == 2 or layout == 4:
#             for r in range(len(verts2)):
#                 for i in range(len(verts2[r])):
#                     theta = i / len(verts2[r])
#                     if layout == 1:
#                         x = theta
#                         y = r
#                     else:
#                         x = math.cos(math.pi * 2 * theta) * r
#                         y = math.sin(math.pi * 2 * theta) * r
#                     pos[verts2[r][i]] = (x, y)
#
#         # elif layout == 5:
#         #     verts, edges = node.to_lists()
#
#         # Create networkxs graph
#         fig, ax = plt.subplots()
#         G = nx.MultiDiGraph()
#         G.add_nodes_from(range(len(verts)))
#         G.add_edges_from(edges)
#         G.nodes(data=True)
#
#         # pos = nx.kamada_kawai_layout(G)
#         # pos = nx.spring_layout(G)
#         # pos = nx.spectral_layout(G)
#         # pos = nx.arf_layout(G)
#         # pos = nx.planar_layout(G)
#
#         nx.draw_networkx_nodes(
#             G,
#             pos,
#             nodelist=range(len(verts)),
#             node_color='tab:blue',
#             node_size=600,
#         )
#         nx.draw_networkx_labels(
#             G,
#             pos,
#             labels = {key: str(node.value) for key,node in enumerate(verts)},
#             font_color="whitesmoke",
#             font_size=10,
#         )
#         nx.draw_networkx_edges(
#             G,
#             pos,
#             arrowstyle="->",
#             arrowsize=20,
#             # edge_color = range(G.number_of_edges()),
#             # edge_cmap = plt.cm.gist_rainbow,
#             width=2,
#             alpha=0.5,
#         )
#         plt.title(title)
#         plt.show()


def plot_graph(node, scale=1, title=None, suptitle=None, **kwargs):
    """Plot the node as a tree"""
    node.reset_index()
    verts, edges = node.to_lists()
    nodes = node.nodes()
    pos = sorted([(n.temp_index ,n.depth()) for n in nodes])

    # Create networkxs graph
    fig, ax = plt.subplots()
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
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
        node_size=600*scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels = {key: vert for key,vert in enumerate(verts)},
        font_color="whitesmoke",
        font_size=10*scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=20*scale,
        # edge_color = range(G.number_of_edges()),
        # edge_cmap = plt.cm.gist_rainbow,
        width=2*scale,
        alpha=0.5,
    )

    # suptitle = f'${node.latex()}$'
    # title, suptitle = suptitle, title
    # plt.suptitle(suptitle)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.title(title)
    if 'result_fitness_func' in kwargs:
        plt.legend(title=f'Fitness = {kwargs['result_fitness_func']([node], **kwargs)[0]}')
    plt.xlabel('Traversal Order')
    plt.ylabel('Depth')
    plt.show()


#
# Control
#

def get_best(all_pops, all_fits, gen=-1, **kwargs):
    """Get the best result of the given run and gen"""
    nodes = []
    # Iterate over all runs
    for run in range(all_pops.shape[0]):
        i = all_fits[run,slice(None),gen,:].argmin()
        node = all_pops[run,slice(None),gen,:].flatten()[i]
        nodes.append(node)
    return nodes


def plot_results(all_pops, all_fits, **kwargs):
    """Plot all standard plots"""

    plot_min_fit(all_pops, all_fits, title='', **kwargs)

    # Plot best
    best = get_best(all_pops, all_fits, **kwargs)
    plot_nodes(best, **kwargs)

    plot_size(all_pops, all_fits, **kwargs)
    plot_quality_gain(all_pops, all_fits, **kwargs)
    plot_success_rate(all_pops, all_fits, **kwargs)
    plot_effective(all_pops, all_fits, **kwargs)
    # plot_noop_size(all_pops, all_fits, **kwargs)


    if len(kwargs['domains']) == 1:
        for i, node in enumerate(best):
            title = 'Best Graph (' + kwargs['test_kwargs'][i + 1][0] + ')'
            plot_graph(node, title=title, **kwargs)
    else:
        table_best(all_pops, all_fits, title='Best Overall', **kwargs)



if __name__ == '__main__':
    name = 'noop'
    all_pops, all_fits, kwargs = load_all(name)
    plot_results(all_pops, all_fits, **kwargs)