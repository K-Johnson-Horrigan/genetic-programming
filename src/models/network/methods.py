"""
Genetic programming functions specifically for the evolution of completed graph models.
"""
import numpy as np
from networkx.classes import subgraph

# from src.models import plot_network


#
# Problem Generation
#

def regular_topology(network_shape, **kwargs):
    """Returns the nodes and links for a regular grid topology"""
    nodes = []
    links = []
    for i in range(network_shape[0]):
        for j in range(network_shape[1]):
            node_id = len(nodes)
            nodes.append((i,j))
            if i > 0:
                node_down = node_id - network_shape[0]
                links.append((node_down, node_id))
            if j > 0:
                node_left = node_id - 1
                links.append((node_left, node_id))
    return nodes, links

def setup(nodes, links, **kwargs):
    """Saves problem specific constants to the kwargs for future calculations, such as fitness"""
    # Array of nodes as [x,y]
    nodes = np.array(nodes)
    kwargs['nodes'] = nodes

    # List of all links as (s1,s2)
    links = tuple(tuple(l) for l in links)
    kwargs['links'] = links

    # Adj matrix of all links_adj
    links_adj = np.zeros((len(nodes), len(nodes)))
    for i, j in links:
        links_adj[i, j] = 1
        links_adj[j, i] = 1
    kwargs['links_adj'] = links_adj

    # List of all interfering links as (e1,e2)
    # Future calculation will usually only be done for links that are known to interfere instead of every link
    interf = []
    for e1 in range(len(links)):
        for e2 in range(e1 + 1, len(links)):
            interf.append((e1,e2))
    kwargs['interf'] = np.array(interf)

    # Distances between all interfering links
    dists = []
    # Minimum number of channels required for separation
    min_c_seps = []
    for e1,e2 in interf:
        s1 = links[e1]
        s2 = links[e2]
        # Calculate distance
        d00 = np.linalg.norm(s1[0] - s2[0])
        d01 = np.linalg.norm(s1[0] - s2[1])
        d10 = np.linalg.norm(s1[1] - s2[0])
        d11 = np.linalg.norm(s1[1] - s2[1])
        dist = min([d00, d01, d10, d11])
        dists.append(dist)
        # Calculate separation
        min_c_sep = min([c for c in range(6) if dist >= kwargs['i_c'][c]])
        min_c_seps.append(min_c_sep)
    kwargs['dists'] = np.array(dists)
    kwargs['min_c_seps'] = np.array(min_c_seps)

    kwargs['interf_mask'] = [True] * len(kwargs['interf'])

    # Setup node ordering
    bfs_nodes = [0]
    bfs_links = []
    for node in bfs_nodes:
        for next_node in range(len(kwargs['nodes'])):
            link = tuple(sorted((node, next_node)))
            # A link exists between the two nodes and has not already been traversed
            if link in kwargs['links'] and link not in bfs_links:
                bfs_links.append(link)
                if next_node not in bfs_nodes:
                    bfs_nodes.append(next_node)
    bfs_links = np.array(bfs_links)
    bfs_links = tuple([(int(u), int(v)) for u, v in bfs_links])
    kwargs['bfs_map'] = [links.index(link) for link in bfs_links]
    kwargs['bfs_demap'] = [bfs_links.index(link) for link in links]

    return kwargs


#
# Initialization Functions
#

def random_network(**kwargs):
    """Returns random network weights"""
    org = []
    for e1,e2 in kwargs['links']:
        org.append(kwargs['rng'].choice(kwargs['channels']))
    org = np.array(org)
    return org


#
# Fitness Functions
#

def remove_node(node_index, interf_mask, **kwargs):
    # if 'interf_mask' not in kwargs[]
    for link in kwargs['links']:
        if node_index in link:
            for i,interf in enumerate(kwargs['interf']):
                if link in interf:
                    interf_mask[i] = False
    return interf_mask


def total_interference(pop, **kwargs):
    """Fitness function based on the total interference"""
    fits = np.empty(len(pop))
    for i,org in enumerate(pop):
        fit = org[kwargs['interf'][:,0]] - org[kwargs['interf'][:,1]]
        fit = np.abs(fit)
        fit = fit >= kwargs['min_c_seps']
        fit = ~fit
        fit = fit * kwargs['interf_mask']
        fit = np.sum(fit)
        fits[i] = fit
    fits = np.array(fits)
    return fits

def num_channel(pop, **kwargs):
    """Fitness function based on max interference"""
    num_channels = np.empty(len(pop))
    for i,org in enumerate(pop):
        num_channels[i] = len(np.unique(org))
    num_channels = np.array(num_channels)
    return num_channels

def max_interference(pop, **kwargs):
    """Fitness function based on max interference"""
    fits = np.empty(len(pop))
    for i,org in enumerate(pop):
        fit = np.abs(org[kwargs['interf'][:,0]] - org[kwargs['interf'][:,1]]) # distances
        fit = kwargs['min_c_seps'] - fit # if 0 or negative, distance is sufficient to avoid interference
        best = np.max(fit)
        fits[i] = best if best > 0 else 0
    return(np.array(fits))

def multi_obj_nsga_esque(pop, **kwargs): # not perfect nsga
    """Fitness function based on the total and maximum interference"""

    obj_1 = total_interference(pop, **kwargs)
    obj_2 = max_interference(pop, **kwargs)
    ranks = np.empty(len(pop))
   
    fronts = [[]]
    dominated_sets = []
    domination_counters = np.empty(len(pop))

    for i_p in range(len(pop)):
        ds = []
        domination_counter = 0

        for i_q in range(len(pop)):
            if((obj_1[i_p] <= obj_1[i_q] and obj_2[i_p] < obj_2[i_q]) or \
               (obj_1[i_p] < obj_1[i_q] and obj_2[i_p] <= obj_2[i_q])): # minimize
               ds.append(i_q) # p dominates q
            elif((obj_1[i_q] <= obj_1[i_p] and obj_2[i_q] < obj_2[i_p]) or \
                 (obj_1[i_q] < obj_1[i_p] and obj_2[i_q] <= obj_2[i_p])):
                domination_counter += 1 # q dominates p

        if domination_counter == 0: # p is undominated
            ranks[i_p] = 1 # p is a 1-st rank solution
            fronts[0].append(i_p)

        dominated_sets.append(ds)
        domination_counters[i_p] = domination_counter

    i = 0
    while fronts[i]:
        Q = []
        for i_p in fronts[i]:
            for i_q in dominated_sets[i_p]:
                domination_counters[i_q] -= 1
                if(domination_counters[i_q] == 0):
                    ranks[i_q] = i + 2
                    if i_q not in Q: Q.append(i_q)
        i += 1
        if(i == len(fronts)):
            fronts.append(Q)

    return np.array(ranks)

# def total_interference_removed_links(pop, **kwargs):
#     """Fitness function based on the total interference"""
#     fits = np.empty(len(pop))
#     for i,org in enumerate(pop):
#         fit = org[kwargs['interf'][:,0]] - org[kwargs['interf'][:,1]]
#         fit = np.abs(fit)
#         fit = fit >= kwargs['min_c_seps']
#         fit = np.sum(fit == False)
#         fits[i] = fit
#     fits = np.array(fits)
#     return fits
#

# def modified_total_interference(pop, **kwargs):
#     """Fitness function based on the total interference"""
#     fits = np.empty(len(pop))
#     for i,org in enumerate(pop):
#         fit = org[kwargs['interf'][:,0]] - org[kwargs['interf'][:,1]]
#         fit = np.abs(fit)
#         fit[fit >= kwargs['min_c_seps']] = 0
#         # fit = np.minimum(fit - kwargs['min_c_seps'], 0)
#         fit = np.sum(fit)
#         fit = 1 / fit
#         fits[i] = fit
#     fits = np.array(fits)
#     return fits


#
# Crossover Functions
#

def one_point_crossover(a, b, **kwargs):
    min_len = len(a)
    max_len = len(a)
    a = list(a)
    b = list(b)
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_max = min(cut_a + len(b) - min_len, cut_a - len(a) + max_len)
    cut_b_min = max(cut_a + len(b) - max_len, cut_a - len(a) + min_len)
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    new_a = np.array(new_a)
    new_b = np.array(new_b)
    return new_a, new_b


def two_point_crossover(a, b, **kwargs):
    min_len = len(a)
    max_len = len(a)
    a = list(a)
    b = list(b)
    # Difference in lengths of the sections to be swapped
    # diff_diff_cuts = len(a) - len(b)
    # kwargs['min_len'] <= len(a) + diff_diff_cuts <= kwargs['max_len']
    # kwargs['min_len'] <= len(b) - diff_diff_cuts <= kwargs['max_len']
    diff_diff_cuts_min = max(min_len - len(a), len(b) - max_len)
    diff_diff_cuts_max = min(max_len - len(a), len(b) - min_len)
    diff_diff_cuts = kwargs['rng'].integers(diff_diff_cuts_min, diff_diff_cuts_max + 1)
    # The length of a cut cannot be negative
    # 0 <= diff_cuts_a <= len(a)
    # 0 <= diff_cuts_a + diff_diff_cuts <= len(b)
    diff_cuts_a = kwargs['rng'].integers(max(0, -diff_diff_cuts), min(len(a), len(b) - diff_diff_cuts) + 1)
    diff_cuts_b = diff_cuts_a + diff_diff_cuts
    cut_a_0 = kwargs['rng'].integers(0, len(a) - diff_cuts_a + 1)
    cut_b_0 = kwargs['rng'].integers(0, len(b) - diff_cuts_b + 1)
    cut_a_1 = cut_a_0 + diff_cuts_a
    cut_b_1 = cut_b_0 + diff_cuts_b
    # Swap the two sections
    new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
    new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
    # assert kwargs['min_len'] <= len(new_a) <= kwargs['max_len']
    # assert kwargs['min_len'] <= len(new_b) <= kwargs['max_len']
    new_a = np.array(new_a)
    new_b = np.array(new_b)
    return new_a, new_b


def bfs_two_point_crossover(a, b, **kwargs):
    """Returns two children from crossover ordered by BFS"""
    new_a = a[kwargs['bfs_map']]
    new_b = b[kwargs['bfs_map']]
    new_a, new_b = two_point_crossover(new_a, new_b, **kwargs)
    new_a = new_a[kwargs['bfs_demap']]
    new_b = new_b[kwargs['bfs_demap']]
    return new_a, new_b


def subgraph_crossover(a, b, **kwargs):
    """"""
    subgraph = subgraph_selection(**kwargs)
    new_a = a.copy()
    new_a[subgraph] = b[subgraph]
    new_b = b.copy()
    new_b[~subgraph] = a[~subgraph]
    new_a = np.array(new_a)
    new_b = np.array(new_b)
    return new_a, new_b


def subgraph_selection(**kwargs):
    """Returns a Boolean array representing if a link is in the subgraph.
    This can then be used to index the array of links."""

    # node_dists = np.zeros((len(kwargs['nodes']),len(kwargs['nodes'])))
    # for i,j in kwargs['links']:
    #     ni = kwargs['nodes'][i]
    #     nj = kwargs['nodes'][j]
    #     d = np.linalg.norm(nj-ni)
    #     node_dists[i, j] = d
    #     node_dists[j, i] = d

    # Boolean list representing if each link is in the subgraph
    subgraph = np.zeros((len(kwargs['links']),), bool)

    # Randomly selected node
    init_node = int(kwargs['rng'].integers(len(kwargs['nodes'])))

    # List of the nodes in the order they are visited
    nodes = [init_node]

    for node in nodes:
        # print('node', node)
        for next_node in range(len(kwargs['nodes'])):
            link = tuple(sorted((node, next_node)))
            # A link exists between the two nodes and has not already been traversed
            if link in kwargs['links']:
                link_index = kwargs['links'].index(link)
                # Continue if subgraph has already been added
                if subgraph[link_index]:
                    continue
                # Probability to add link to subgraph
                elif kwargs['rng'].random() < kwargs['subgraph_crossover_p_branch']:
                    # print('\tADDED', link)
                    subgraph[link_index] = True
                    # Add next npde to list of nodes to visit
                    if next_node not in nodes:
                        nodes.append(next_node)

    return subgraph


#
# Mutation Functions
#

def point_mutation(org, **kwargs):
    """Randomly change a value in a random line"""
    org_copy = org.copy()
    index = kwargs['rng'].integers(len(org))
    org_copy[index] = kwargs['rng'].choice(kwargs['channels'])
    return org_copy

#
# Debug
#

if __name__ == '__main__':

    # Testing setup
    n, l = regular_topology((2,2))
    kwargs = setup(n,l, i_c=[2, 1.125, 0.75, 0.375, 0.125, 0])
    kwargs['rng'] = np.random.default_rng()
    print(kwargs)

    interf_mask = [True] * len(kwargs['interf'])
    interf_mask = remove_node(0, **kwargs)


    print()
    print(kwargs['interf'])
    print(interf_mask)

    # s = subgraph_selection(**kwargs)
    # l = [link for i,link in enumerate(kwargs['links']) if s[i]]
    # print(l)
    # plot_network(show=True, save=False, nodes=n, links=l)

    pass