"""
Genetic programming functions specifically for the evolution of linear models.
"""
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


#
# Setup Functions
#

def setup_uniform_clients(**kwargs):
    """Uses the seed to generate the client positions and returns the updated kwargs"""
    rng = np.random.default_rng(kwargs['seed'])
    clients = rng.uniform(kwargs['min_value'], kwargs['max_value'], size=(kwargs['num_clients'], 2))
    kwargs['clients'] = clients
    return kwargs


#
# Initialization Functions
#

def random_uniform_router_coords(**kwargs):
    """Generate a random list of coords"""
    return kwargs['rng'].uniform(kwargs['min_value'], kwargs['max_value'], size=(kwargs['num_routers'],2))



#
# Fitness Functions
#

def coverage_arr(routers, **kwargs):
    """Returns an array with a value for each client with the index of the router covering it, -1 is used for uncovered clients"""
    cov_arr = [-1] * len(kwargs['clients'])
    for i,client in enumerate(kwargs['clients']):
        for j,router in enumerate(routers):
            dist = np.linalg.norm(client - router)
            if dist < kwargs['radius']:
                cov_arr[i] = j
                break
    return cov_arr


def router_adj_mat(routers, **kwargs):
    """Returns the adjacency matrix of the routers"""
    adj_mat = np.zeros((len(routers),len(routers)), int)
    for i,r0 in enumerate(routers):
        for j,router in enumerate(routers):
            if i != j:
                dist = np.linalg.norm(r0 - router)
                if dist <= 2 * kwargs['radius']:
                    adj_mat[i,j] = 1
                    adj_mat[j,i] = 1
    return adj_mat

def num_connected(adj_mat, **kwargs):
    """Returns the number of connected components given the adjacency matrix of routers"""
    return connected_components(
        csgraph=csr_matrix(adj_mat),
        directed=False,
        return_labels=False
    )



def cov_con_sum_fitness(pop, **kwargs):
    """Calculate the fitness for each organism based on the coverage minus connectedness"""
    # if 'clients' not in kwargs: kwargs = kwargs['setup_func'](**kwargs)

    n = len(kwargs['clients'])
    m = kwargs['num_routers']
    fits = np.empty(len(pop))
    for k,routers in enumerate(pop):
        n_cov = sum(np.array(coverage_arr(routers, **kwargs)) != -1)
        n_con = num_connected(router_adj_mat(routers, **kwargs))
        n_cov_norm = n_cov/n
        n_con_norm = (m-n_con)/(m-1)
        fits[k] = (n_cov_norm + n_con_norm) / 2
    fits = np.array(fits)
    return fits


def cov_con_entropy_fitness(pop, **kwargs):
    """Calculate the fitness for each organism based on the coverage minus connectedness"""
    # if 'clients' not in kwargs: kwargs = kwargs['setup_func'](**kwargs)

    # Constants from paper
    n = len(kwargs['clients'])
    m = kwargs['num_routers']

    fits = np.empty(len(pop))
    for k,routers in enumerate(pop):

        # Number of covered clients in j-th router
        n_j = np.zeros(len(routers))
        for i, router in enumerate(routers):
            for j, client in enumerate(kwargs['clients']):
                dist = np.linalg.norm(client - router)
                if dist <= kwargs['radius']:
                    n_j[i] += 1

        # Probability of coverage
        P_i = n_j / n

        # Coverage entropy
        H_cov_0 = np.log(P_i)
        H_cov_1 = np.nan_to_num(H_cov_0)
        H_cov_2 = P_i * H_cov_1
        H_cov_3 = sum(H_cov_2)
        H_cov_4 = np.log(m)
        H_cov_5 = H_cov_3 / H_cov_4
        H_cov = -H_cov_5
        # H_cov = - sum(P_i * np.nan_to_num(np.log(P_i))) / np.log(m)

        # Construct graph adj matrix from routers and clients
        G = nx.from_numpy_array(router_adj_mat(routers, **kwargs))
        G.add_nodes_from(range(kwargs['num_routers'], kwargs['num_routers'] + kwargs['num_clients']))
        client_edges = coverage_arr(routers, **kwargs)
        client_edges = tuple([(i + kwargs['num_routers'], j) for i, j in enumerate(client_edges) if j != -1])
        G.add_edges_from(client_edges)
        adj = nx.to_numpy_array(G)
        G_n, G_labels = connected_components(csgraph=csr_matrix(adj), directed=False,return_labels=True)

        # Construct graph adj matrix from routers
        # G_n, G_labels = connected_components(csgraph=csr_matrix(router_adj_mat(routers, **kwargs)), directed=False, return_labels=True)

        # Size of the i-th sub-network
        _, G_i_size = np.unique(G_labels, return_counts=True, equal_nan=False)

        # Probability of connectivity
        P_j = G_i_size / (n + m)

        # Connectivity entropy
        if G_n > 1:
            H_con_0 = np.log(P_j)
            H_con_1 = np.nan_to_num(H_con_0)
            H_con_2 = P_j * H_con_1
            H_con_3 = sum(H_con_2)
            H_con_4 = np.log(G_n)
            H_con_5 = H_con_3 / H_con_4
            H_con = -H_con_5
            # H_con = - sum(P_j * np.log(P_j)) / np.log(G_n)
        else:
            H_con = 0

        fits[k] = H_cov - H_con

    fits = np.array(fits)
    return fits


def lexicase_fitness(pop, **kwargs):
    """Calculate the fitness for each organism based on its lexicase selection ranking"""

    # loop: 
    fits = np.full(len(pop), -1)
    cov_vals = np.empty((len(pop),2))
    con_vals = np.empty((len(pop),2))

    for k,routers in enumerate(pop):
        cov_vals[k][1] = sum(np.array(coverage_arr(routers, **kwargs)) != -1)
        con_vals[k][1] = num_connected(router_adj_mat(routers, **kwargs))
        cov_vals[k][0] = con_vals[k][0] = k
    
    con_vals = con_vals[con_vals[:, 1].argsort()]
    cov_vals = cov_vals[cov_vals[:, 1].argsort()]

    cur_fitness = len(pop)
    con_i = cov_i = len(pop) - 1
    while(con_i >= 0 and cov_i >= 0):
        # pick a random test case to go first:
        if (np.random.rand() < 0.5): # coverage first
            best = cov_vals[cov_i][1]
            while(cov_i >= 0 and cov_vals[cov_i][1] >= best - kwargs['epsilon']):
                if(fits[int(cov_vals[cov_i][0])] == -1): 
                    fits[int(cov_vals[cov_i][0])] = cur_fitness
                cov_i -= 1
            
        else: # connectivity first
            best = con_vals[con_i][1]
            while(con_i >= 0 and con_vals[con_i][1] >= best - kwargs['epsilon']):
                if(fits[int(con_vals[con_i][0])] == -1): 
                    fits[int(con_vals[con_i][0])] = cur_fitness
                con_i -= 1
        cur_fitness -= 1
    
    fits = np.array(fits)
    return fits
    

# 
# Extra data collection
#
def get_con_vals(pop, **kwargs):
    con_vals = np.empty(len(pop))
    for k,routers in enumerate(pop):
        con_vals[k] = num_connected(router_adj_mat(routers, **kwargs))
    return con_vals

def get_cov_vals(pop, **kwargs):
    cov_vals = np.empty(len(pop))
    for k,routers in enumerate(pop):
        cov_vals[k] = sum(np.array(coverage_arr(routers, **kwargs)) != -1)
    return cov_vals


#
# Crossover Functions
#

def coords_one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_min = max(cut_a + len(b) - kwargs['max_len'], cut_a - len(a) + kwargs['min_len'])
    cut_b_max = min(cut_a + len(b) - kwargs['min_len'], cut_a - len(a) + kwargs['max_len'])
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    return new_a, new_b

def coords_two_point_crossover(a, b, **kwargs):
    """Modified crossover to work for fixed lengths"""
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


def coords_fixed_two_point_crossover(a, b, **kwargs):
    a = list(a)
    b = list(b)
    cut_0 = kwargs['rng'].integers(0, len(a)-1)
    cut_1 = kwargs['rng'].integers(0, len(a))
    if cut_0 > cut_1: cut_0, cut_1 = cut_1, cut_0
    # Swap the two sections
    new_a = a[:cut_0] + b[cut_0:cut_1] + a[cut_1:]
    new_b = b[:cut_0] + a[cut_0:cut_1] + b[cut_1:]
    return new_a, new_b


def coords_reorder_crossover(a, b, **kwargs):

    adjusted_b = np.array(b.copy())
    new_b = np.empty_like(b)

    for i in range(len(a)):
        dists = np.linalg.norm(adjusted_b - a[i], axis=1)
        min_dist_index = np.argmin(dists)
        new_b[i] = b[min_dist_index]
        adjusted_b[min_dist_index] = 1000000

    return coords_fixed_two_point_crossover(a, new_b, **kwargs)


#
# Mutation Functions
#

def coords_point_mutation(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    # Replace the argument
    org[index] = kwargs['rng'].uniform(kwargs['min_value'], kwargs['max_value'])
    return org

def coords_point_mutation_2d(org, **kwargs):
    """Randomly change a value in a random line"""
    # Select a random value
    index = kwargs['rng'].integers(len(org))
    # Replace the argument
    # org[index] = point_mutation(org[index], **kwargs)
    coords_point_mutation(org[index], **kwargs)
    return org


#
# Debug
#

if __name__ == '__main__':
    pass

    # l = 6
    # a = np.random.random((l, 2))
    # b = np.random.random((l, 2))
    #
    # print('a')
    # print(a)
    # print('b')
    # print(b)
    #
    # adjusted_b = b.copy()
    # new_b = np.empty_like(b)
    #
    # inds = []
    #
    # print('calc')
    # for i in range(len(a)):
    #     dists = np.linalg.norm(adjusted_b - a[i], axis=1)
    #     # print(dists)
    #     min_dist_index = np.argmin(dists)
    #     # inds.append(min_dist_index)
    #     new_b[i] = b[min_dist_index]
    #     adjusted_b[min_dist_index] = 100000
    #
    #
    # print(inds)
    # print('new')
    # print(new_b)
    #
    #
    # # c = np.abs(a-b)
    #
    # # d = np.linalg.norm(a - b, axis=1)
    # # print(d)
    # #
    # # i = np.argmin(np.linalg.norm(a - b, axis=1))
    # #
    # # print(i)
    # # router_adj_mat(a, radius=.2)
    #
    # plt.scatter(*a.T, color='blue', marker='d')
    # plt.scatter(*b.T, color='orange', marker='s')
    # for i in range(len(a)):
    #     x = (a[i][0], new_b[i][0])
    #     y = (a[i][1], new_b[i][1])
    #     plt.plot(x,y)
    # plt.show()

