"""
Genetic programming functions specifically for the evolution of completed graph models.
"""
import numpy as np

#
# Problem Generation
#

def regular_topology(shape):

    nodes = []
    links = []

    for i in range(shape[0]):
        for j in range(shape[1]):

            node_id = len(nodes)
            nodes.append((i,j))

            if i > 0:
                node_down = node_id - shape[0]
                links.append((node_down, node_id))

            if j > 0:
                node_left = node_id - 1
                links.append((node_left, node_id))



    return nodes, links


def setup(nodes, links, **kwargs):

    # List of nodes as [x,y]
    nodes = np.array(nodes)
    kwargs['nodes'] = nodes

    # List of all links as [node_1, node_2]
    links = tuple(tuple(l) for l in links)
    kwargs['links'] = links

    # Adj matrix of all links_adj
    links_adj = np.zeros((len(nodes), len(nodes)))
    for i, j in links:
        # i,j = sorted((i,j))
        links_adj[i, j] = 1
        links_adj[j, i] = 1
    kwargs['links_adj'] = links_adj

    # List of all interfering links as [s1,s2]
    interf = []
    for s1 in range(len(links)):
        for s2 in range(s1 + 1, len(links)):
            interf.append((s1,s2))
    kwargs['interf'] = np.array(interf)

    # Distances between all interfering links
    # dists = -np.ones((len(links_list), len(links_list)))
    dists = []
    min_c_seps = []
    for e1,e2 in interf:
        s1 = links[e1]
        s2 = links[e2]
        d00 = np.linalg.norm(s1[0] - s2[0])
        d01 = np.linalg.norm(s1[0] - s2[1])
        d10 = np.linalg.norm(s1[1] - s2[0])
        d11 = np.linalg.norm(s1[1] - s2[1])
        dist = min([d00, d01, d10, d11])
        # dists[s1, s2] = dist
        # dists[s2, s1] = dist
        dists.append(dist)

        min_c_sep = min([c for c in range(6) if dist >= kwargs['i_c'][c]])
        min_c_seps.append(min_c_sep)
    kwargs['dists'] = np.array(dists)
    kwargs['min_c_seps'] = np.array(min_c_seps)

    return kwargs

#
# Initialization Functions
#

def random_network(**kwargs):
    org = []
    for link in kwargs['interf']:
        # l = tuple(sorted(link))
        # org[l] = kwargs['rng'].choice(kwargs['channels'])
        org.append(kwargs['rng'].choice(kwargs['channels']))
    org = np.array(org)
    return org

    # adj = np.zeros((len(kwargs['links']),len(kwargs['links'])))
    # for link in kwargs['links']:
    #     l = tuple(sorted(link))
    #     adj[l[0],l[1]] = kwargs['rng'].choice(kwargs['channels'])
    # return org

#
# Fitness Functions
#

def total_interference(pop, **kwargs):
    fits = np.empty(len(pop))
    for i,org in enumerate(pop):
        fit = org[kwargs['interf'][:,0]] - org[kwargs['interf'][:,1]]
        fit = np.abs(fit)
        fit = fit >= kwargs['min_c_seps']
        fit = np.sum(fit == False)
        fits[i] = fit
    fits = np.array(fits)
    return fits

#
# Crossover Functions
#

def one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_min = max(cut_a + len(b) - kwargs['max_len'], cut_a - len(a) + kwargs['min_len'])
    cut_b_max = min(cut_a + len(b) - kwargs['min_len'], cut_a - len(a) + kwargs['max_len'])
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
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
    pass