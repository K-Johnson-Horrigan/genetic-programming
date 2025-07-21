"""All functions relevant to plotting."""

import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.genetics import Linear, run_self_rep
from src.genetics.tmgp import _run_maze_tm, maze_fitness
from src.utils.save import load_kwargs, load_runs
from src.utils.utils import cartesian_prod


#
# Plotting For Evolution Based Classes
#

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
    plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    plt.show()


def plot_tm_graph(trans, ax=None, scale=1, title=None, save=True, show=True, **kwargs):
    """Plot a TM as a graph"""

    if ax is None:
        fig, ax = plt.subplots()

    # Convert trans array into a list of transitions
    shape = trans.shape
    states = range(shape[0])
    symbols = range(shape[1])
    trans = [[state, symbol, *trans[(state, symbol)]] for state in states for symbol in symbols]

    # Convert a list of transitions into a list of vertices and edges
    verts = []
    edges = []
    edge_labels = {}
    for transition in trans:
        state0, symbol0, state1, symbol1, *move = transition
        # Add vertices and edges if they are not already present
        if state0 not in verts: verts.append(state0)
        if state1 not in verts: verts.append(state1)
        edge = (verts.index(state0), verts.index(state1))
        if edge not in edges:
            edges.append(edge)
        # Either create a label or append to an existing label
        edge_label = f'{symbol0}→{symbol1} (' + ','.join(map(str,move)) + ')'
        if edge in edge_labels:
            edge_labels[edge] += '\n' + edge_label
        else:
            edge_labels[edge] = edge_label

    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)
    connectionstyle = [f"arc3,rad={r}" for r in [.25, .75]]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=range(len(verts)),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="-|>",
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
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        connectionstyle=connectionstyle,
        # edge_labels = {edges[key]: label for key,label in enumerate(edge_labels)},
        edge_labels=edge_labels,
        alpha=0.5,
        # label_pos=0.0,
        # node_size=24000 * scale,
        bbox=None,
    )
    plt.title(title)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


def plot_tm_maze(trans, ax=None, title=None, save=True, show=True, **kwargs):
    """Plot the resulting tape after running the TM"""

    if ax is None:
        fig, ax = plt.subplots()

    tm = _run_maze_tm(trans, **kwargs)
    fit = maze_fitness([trans], **kwargs)[0]
    tape = tm.get_tape_as_array()

    colors = tm.state_history
    xy = list(zip(*tm.head_pos_history))[::-1]

    ax.set_title(f'{title} ({fit})')
    ax.plot(*xy)
    ax.scatter(*xy, c=colors)
    ax.imshow(tape)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


def plot_tm_trans_array(trans, ax=None, title=None, save=True, show=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # States and symbols can be inferred instead of using kwargs
    shape = trans.shape
    states = list(range(shape[0]))
    symbols = list(range(shape[1]))
    im = trans[:,:,1]
    # Append states
    im = np.concat(([list(symbols)], im), axis=0)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('States')
    ax.set_xlabel('Symbols')
    # ax.set_axis_off()
    ax.invert_yaxis()
    ax.set_xlim((-1.5, shape[1]-.5))
    # ax.set_ylim((-1.5, shape[0]+.5))
    for state in states:
        ax.text(-1, state+1, state, ha='center', va='center', color='k')
    for symbol in symbols:
        ax.text(symbol, 0, symbol, ha='center', va='center', color='k')
    # Scatter plot points
    # Initial values are used for the states axis
    xs = [-1] * shape[0]
    ys = [y+1 for y in states]
    c = states.copy()
    # Loop over data dimensions and create text annotations.
    for state in states:
        for symbol in symbols:
            xs.append(symbol)
            ys.append(state+1)
            c.append(trans[state, symbol, 0])
    # marker = ((-1,-1),(1,-1),(-1,1))
    marker = 'o'
    ax.scatter(xs, ys, marker='o', s=1000, c=c, edgecolors='black')
    ax.plot((-.5,-.5,shape[1]-.5),(shape[0]+.5,.5,.5), color='black')
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


#
# Data Based Plotting
#

def plot_fitness(all_pops, all_fits, ax=None, save=True, show=True, **kwargs):
    """Plot the average of the runs' minimum fitness for each test"""
    if ax is None:
        fig, ax = plt.subplots()
    x = np.array(range(all_fits.shape[2]))
    for test in range(all_fits.shape[0]):
        if kwargs['minimize_fitness']:
            y = np.mean(np.min(all_fits[test], axis=2), axis=0)
            ax.set_ylabel('Average Min Fitness Value')
        else:
            y = np.mean(np.max(all_fits[test], axis=2), axis=0)
            ax.set_ylabel('Average Max Fitness Value')
        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
        # y_std = np.std(np.min(all_fits[test], axis=2), axis=0)
        # ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)
        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)
    # ax.set_yscale('log')
    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/Fitness.png')
    if show:
        plt.show()
    plt.close()


def plot_means(values, ylabel, ax=None, save=True, show=True, **kwargs):
    """Plot the means of some values"""
    if ax is None:
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
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    if show:
        plt.show()
    plt.close()


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
    plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    plt.show()


def plot_box(values, ylabel, ax=None, save=True, show=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    positions = range(len(kwargs['test_kwargs']) - 1)
    ys = [values[test].ravel() for test in range(len(kwargs['test_kwargs']) - 1)]
    # ax.boxplot(
    ax.violinplot(
        ys,
        positions=positions,
        # patch_artist=True,
        showmeans=False,
        showmedians=True,
    )
    ax.yaxis.grid(True)
    ax.set_xticks(ticks=positions, labels=[test[0] for test in kwargs['test_kwargs'][1:]])
    ax.set_xlabel(kwargs['test_kwargs'][0][0])
    ax.set_ylabel(ylabel)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
    if show:
        plt.show()
    plt.close()


# def plot_hist(values, ylabel):
#     fig, ax = plt.subplots()
#     for test in range(values.shape[0]):
#         label = kwargs['test_kwargs'][test + 1][0]
#         xs = values[test, :, -1].ravel()
#         # ax.boxplot(xs,
#         #     positions=[test],
#         #     label=label,
#         #     patch_artist=True,
#         #     # showmeans=False,
#         #     # showfliers=False,
#         #     # medianprops={"color": "white", "linewidth": 0.5},
#         #     # boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
#         #     # whiskerprops={"color": "C0", "linewidth": 1.5},
#         #     # capprops={"color": "C0", "linewidth": 1.5}
#         # )
#     plt.xlabel('Generation')
#     plt.ylabel(ylabel)
#     plt.legend(title=kwargs['test_kwargs'][0][0])
#     plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{ylabel}.png')
#     plt.show()


#
# Tables
#

# def table_best(all_pops, all_fits, **kwargs):
#     """Plot the best result of the given run and gen"""
#     xs = [np.linspace(*domain) for domain in kwargs['domains']]
#     xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
#     y_true = np.array([[kwargs['target_func'](*list(x))] for x in xs])
#     table = np.concat((xs, y_true), axis=1)
#     # Iterate over all runs
#     for run in range(len(kwargs['test_kwargs']) - 1):
#         i = all_fits[run, :, :, :].argmin()
#         node = all_pops[run, :, :, :].flatten()[i]
#         y_node = [[node(*x)] for x in xs]
#         tab = np.concat((table, y_node), axis=1)
#         print('\n', node, sep='')
#         for row in tab:
#             print(('f(' + ', '.join(['{}'] * len(kwargs['domains'])) + ') = {} | {}').format(*row))


def table_best(obj, **kwargs):
    """Plot the best result of the given run and gen"""
    table = []
    cases = cartesian_prod(*kwargs['domains'])
    y_target = np.array([kwargs['target_func'](*list(xs)) for xs in cases])
    # Calculate values
    y_actual = []
    for case in cases:
        l = Linear([[0] + list(case) + [0], np.ravel(obj)])
        l.run(kwargs['timeout'])
        y_actual = np.append(y_actual, l.vars[-1])

    table.append(range(len(cases)))
    table.append(cases)
    table.append(y_target)
    table.append(y_actual)
    table.append(abs(y_target - y_actual))
    table.append(abs(y_target - y_actual) ** 2)

    for row in zip(*table):
        row = '{:2} │ {} │ {:2} │ {:4.1f} │ {:4.1f} │ {:5.1f} │'.format(*row)
        print(row)

    row = []
    row.append('')
    row.append('')
    row.append('')
    row.append('')
    row.append(sum(abs(y_target - y_actual)))
    row.append(sum((abs(y_target - y_actual)) ** 2))
    row.append(sum((abs(y_target - y_actual)) ** 2) / len(cases))
    row.append((sum((abs(y_target - y_actual)) ** 2) / len(cases)) ** (1 / 2))
    row = '{:2} │ {:5} │ {:2} │ {:4} │ {:4.1f} │ {:5.1f} │ {} │ {} │'.format(*row)
    print(row)


#
# Control
#

def get_best(all_pops, all_fits, gen=-1, **kwargs):
    """Get the best result of the given run and gen"""
    best_objs = []
    best_fits = []
    # Iterate over all runs
    for run in range(all_pops.shape[0]):
        if kwargs['minimize_fitness']:
            i = all_fits[run, slice(None), gen, :].argmin()
        else:
            i = all_fits[run, slice(None), gen, :].argmax()
        best_obj = all_pops[run, slice(None), gen, :].flatten()[i]
        best_fit = all_fits[run, slice(None), gen, :].flatten()[i]
        best_objs.append(best_obj)
        best_fits.append(best_fit)
    return best_objs, best_fits


def plot_grid(all_pops, all_fits, plot_func, title=None, save=True, show=True, **kwargs):
    """Plots a grid of plots over the test kwargs"""
    best = get_best(all_pops, all_fits, **kwargs)
    if len(kwargs['test_kwargs'][0]) < 2:
        print(f'Plotting failed for {plot_func.__name__}')
    else:
        # Values of first test_kwargs
        zipped0 = list(zip(*kwargs['test_kwargs'][1:]))[1]
        array0 = np.empty((len(zipped0),), 'object')
        array0[:] = zipped0
        values0, counts0 = np.unique(array0, return_counts=True)
        # Values of second test_kwargs
        zipped1 = list(zip(*kwargs['test_kwargs'][1:]))[2]
        array1 = np.empty((len(zipped1),), 'object')
        array1[:] = zipped1
        values1, counts1 = np.unique(array1, return_counts=True)
        # Setup grid
        nrows = len(counts0)
        ncols = len(counts1)
        if not ((counts0 == counts0[0]).all() and (counts1 == counts1[0]).all() and nrows * ncols == len(
                kwargs['test_kwargs'][1:])):
            print(f'Plotting failed for {plot_func.__name__}')
        else:
            scale, dpi = 4, 400 # Save resolution
            # scale, dpi = 2, 200
            fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * scale, ncols * scale), dpi=dpi)

        # Plot best results of each test
        for i, tm in enumerate(best):
            print(f'Plotting grid {i+1} of {len(best)} for {plot_func.__name__}')
            plot_func(tm, ax=axs.ravel()[i], title=kwargs['test_kwargs'][i+1][0], show=False, save=False, **kwargs)
        fig.supxlabel(kwargs['test_kwargs'][0][2])
        for col in range(ncols):
            axs[-1, col].set_xlabel(values1[col])
        fig.supylabel(kwargs['test_kwargs'][0][1])
        for row in range(nrows):
            axs[row, 0].set_ylabel(values0[row])
        if save:
            plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
        if show:
            plt.show()
        plt.close()


def plot_results(all_pops, all_fits, **kwargs):
    """Plot all standard plots"""
    path = f'{kwargs["saves_path"]}{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')

    plot_fitness(all_pops, all_fits, show=False, **kwargs)

    plot_means(np.vectorize(lambda x: len(x))(all_pops), 'Average Length', show=False, **kwargs)
    # plot_medians(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')
    # plot_hist(np.vectorize(lambda x: len(x[0]))(all_pops), 'Average Number of Nodes')

    plot_box(all_fits[:,:,-1], 'Final Fitness', show=False, **kwargs)

    # plot_grid(all_pops, all_fits, plot_func=plot_tm_maze, title='Best Solutions', show=False, **kwargs)
    # plot_grid(all_pops, all_fits, plot_func=plot_tm_graph, title='Best Graphs', show=False, **kwargs)
    # plot_grid(all_pops, all_fits, plot_func=plot_trans_array, title='Best Transition Arrays', show=False, **kwargs)
    # plot_grid(all_pops, all_fits, plot_func=plot_fitness, title='Best Solutions', show=False, **kwargs)

    # Plot best results of each test
    bests = zip(*get_best(all_pops, all_fits, **kwargs))
    for i, best in enumerate(bests):
        best_obj, best_fit = best
        test_name = kwargs['test_kwargs'][i+1][0]
        print(f'Test: {test_name}, Fitness: {best_fit}')
        table_best(best_obj, **kwargs)

        l = Linear([[0], np.ravel(best_obj)])

        # l = run_self_rep(code_1d, **kwargs)
        print(l)
        # l = run_self_rep(l.mem[2], **kwargs)
        # print(l)




if __name__ == '__main__':
    # name = 'unstable_self_rep_0'
    name = 'mult_1'
    kwargs = load_kwargs(name, '../../saves/')
    pops, fits = load_runs(**kwargs)
    plot_results(pops, fits, **kwargs)