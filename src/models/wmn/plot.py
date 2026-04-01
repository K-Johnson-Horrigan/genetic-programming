"""All functions relevant to plotting."""

import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, patches

from src.models import router_adj_mat, coverage_arr
from src.models.wmn import cov_con_entropy_fitness
from src.utils.save import load_kwargs, load_pop, load_fits, load_seed, load_pops, load_seeds, load_extra_1, load_extra_2
from src.utils.utils import cartesian_prod

#
# Model Based Plotting
#

def plot_network(routers, ax=None, save=False, show=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    # Draw range of each router
    for router in routers:
        circle = patches.Circle(router, kwargs['radius'], color='g', fill=True, alpha=0.1)
        ax.add_patch(circle)
        circle = patches.Circle(router, kwargs['radius'], color='g', fill=False)
        ax.add_patch(circle)

    # Position of all nodes
    pos = list(routers)

    # Start with the connection between all routers
    G = nx.from_numpy_array(router_adj_mat(routers, **kwargs))

    scale = 1
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=range(len(routers)),
        node_color='green',
        edgecolors='green',
        node_size=120 * scale,
    )

    if 'setup_func' in kwargs:
        # Reconstruct the clients from the setup function
        kwargs = kwargs['setup_func'](**kwargs)

    if 'clients' in kwargs:
        pos += list(kwargs['clients'])

        # Edges between routers and clients
        G.add_nodes_from(range(kwargs['num_routers'], kwargs['num_routers'] + kwargs['num_clients']))
        client_edges = coverage_arr(routers, **kwargs)
        client_edges = tuple([(i + kwargs['num_routers'], j) for i, j in enumerate(client_edges) if j != -1])
        G.add_edges_from(client_edges)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=range(len(routers), len(pos)),
            node_color='red',
            edgecolors='black',
            node_size=60 * scale,
        )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=G.edges,  # Specify edge order
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        alpha=0.5,
        node_size=600 * scale,
    )

    # Add axis labels which are removed by default
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    ax.set_title(save)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{save}.png')
    if show:
        plt.show()
    plt.close()


#
# Data Based Plotting
#

def plot_fitness(all_fits, ax=None, save='Fitness', show=True, **kwargs):
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

        # Error bands
        y_std = np.std(np.max(all_fits[test], axis=2), axis=0)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # Min bands
        # y = np.min(np.min(all_fits[test], axis=2), axis=0)
        # plt.plot(x, y,  ':', label=kwargs['test_kwargs'][test + 1][0]+' (min-min)')

        # y = np.max(np.max(all_fits[test], axis=2), axis=0)
        # plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0]+' (max-max)')

        # y = np.max(np.min(all_fits[test], axis=2), axis=0)
        # plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0]+' (max-min)')

        # for run in range(all_fits[test].shape[0]):
        #     y = np.mean(all_fits[test,run], axis=1)
        #     plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0] + f' ({run})')

        # Scatter plot all points
        # xx = x.reshape((1,len(x),1)).repeat(all_fits.shape[1], axis=0).repeat(all_fits.shape[3], axis=2).ravel()
        # yy = all_fits[test].ravel()
        # plt.scatter(xx, yy, 0.1)

    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{save}.png')
    if show:
        plt.show()
    plt.close()

#
# Control
#

def get_best(all_fits, gen=slice(None), **kwargs):
    """Get the best result of the given run and gen"""
    best_orgs = []
    best_fits = []
    best_indices = []
    best_seeds = []
    for test in range(all_fits.shape[0]):
        if kwargs['minimize_fitness']:
            index = np.unravel_index(all_fits[test,:,gen,:].argmin(), all_fits[test,:,gen,:].shape)
        else:
            index = np.unravel_index(all_fits[test,:,gen,:].argmax(), all_fits[test,:,gen,:].shape)
        # Unpack the index as either including generation or not
        if len(index) == 3:
            run, gen_index, org = index
        else:
            run, org = index
            gen_index = gen
        # Store the best organisms and its fitness
        best_orgs.append(load_pop(test,run,**kwargs)[gen_index,org])
        best_fits.append(all_fits[test,run,gen_index,org])
        best_indices.append([test,run,gen_index,org])
        best_seeds.append(load_seed(test,run, **kwargs))
    return best_orgs, best_fits, best_indices, best_seeds


def plot_results(all_fits, **kwargs):
    """Plot all standard plots"""

    # Setup output to save images
    path = f'{kwargs["saves_path"]}{kwargs["name"]}/plots/'
    os.makedirs(path, exist_ok=True)
    print('Plotting results')
    
    if(kwargs['extra_data']):
        plot_extra_1(load_extra_1(**kwargs), **kwargs)
        plot_extra_2(load_extra_2(**kwargs), **kwargs)
    plot_fitness(all_fits, show=True, **kwargs)
    


    # Recalculate fitness using alternate fitness function
    # all_pops = load_pops(**kwargs)
    # new_fits = np.array(all_fits).copy()
    # seeds = load_seeds(0, **kwargs)
    # for run, seed in enumerate(seeds):
    #     kwargs['seed'] = seed
    #     print(run)
    #     kwargs = kwargs['setup_func'](**kwargs)
    #     for gen in range(all_pops.shape[2]):
    #         new_fits[0,run,gen] = kwargs['fitness_func'](all_pops[0,run,gen], **kwargs)
    # plot_fitness(new_fits, show=True, **kwargs)




    # Iterate over the best individuals of each test
    bests = zip(*get_best(all_fits, **kwargs))
    for i, best in enumerate(bests):
        best_org, best_fit, best_index, best_seed = best
        test_name = kwargs['test_kwargs'][i+1][0]
        kwargs['seed'] = int(best_seed)
        print(f'Best of {test_name}, Fitness {best_fit}')

        # Reconstruct the clients from the setup function
        kwargs = kwargs['setup_func'](**kwargs)

        f = kwargs['fitness_func']([best_org], **kwargs)
        print(f)

        print(best_org)

        title = f'Best of {test_name}, Fitness {best_fit}'
        plot_network(best_org, save=title, **kwargs)


def plot_extra_1(extra_1, **kwargs):
    """Plot the average of the runs' max extra_1 for each test"""
    show=True
    save='extra_data_1'
    fig, ax = plt.subplots()
    x = np.array(range(extra_1.shape[2]))
    for test in range(extra_1.shape[0]):
        y = np.mean(np.max(extra_1[test], axis=2), axis=0)
        ax.set_ylabel('Mean max connectivity')
        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{save}.png')
    if show:
        plt.show()
    plt.close()

def plot_extra_2(extra_2, **kwargs):
    """Plot the average of the runs' max extra_2 for each test"""
    show=True
    save='extra_data_2'
    fig, ax = plt.subplots()
    x = np.array(range(extra_2.shape[2]))
    for test in range(extra_2.shape[0]):
        y = np.mean(np.max(extra_2[test], axis=2), axis=0)
        ax.set_ylabel('Mean max coverage')
        plt.plot(x, y, label=kwargs['test_kwargs'][test + 1][0])
    ax.set_xlabel('Generation')
    plt.legend(title=kwargs['test_kwargs'][0][0])
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{save}.png')
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    name = 'fitness_0'
    kwargs = load_kwargs(name, '../../../saves/placement/')
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)



    # a = np.random.random((8,2))
    # b = np.random.random((8,2))

    # router_adj_mat(a, radius=.2)

    # plot_network(a, radius=.2, save=False)

    # dx = 1
    # dy = 1
    #
    # del kwargs['setup_func']
    # kwargs['clients'] = [
    #     [x*dx,y*dy] for x in [-1,0,1] for y in [-1,0,1]
    # ]
    # kwargs['num_clients'] = len(kwargs['clients'])
    # print(kwargs['clients'])
    # r = np.array([[x * dx, 0] for x in [-1, 0, 1]])
    # kwargs['num_routers'] = len(r)
    #
    # f = kwargs['fitness_func']([r], **kwargs)
    # print(f)
    #
    # plot_network(r, **kwargs)