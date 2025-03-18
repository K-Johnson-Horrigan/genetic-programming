import numpy as np

from node import *
from evolve import *
from plot import *
from utils import save_all

from math import sin, cos

"""Functions relevant to implementing genetic programming"""

#
# Initialization
#

def random_tree(init_tree_depth, ops, terminals, p_branch=0.5, init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and init_tree_depth > 0:
        op = random.choice(ops)
        children = [random_tree(init_tree_depth - 1, ops, terminals, p_branch, False) for _ in range(Node.valid_ops[op])]
        return Node(op, children)
    # Create a leaf
    else:
        return Node(random.choice(terminals))

#
# Evaluation
#

def mse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_true = np.array([target_func(*list(x)) for x in xs])
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        y_node = [node(*x) for x in xs]
        fit = (sum((abs(y_true - y_node)) ** 2) / len(xs)) ** (1/2)
        # fit = sum(abs(y_true - y_node))
        # fits.append(fit)
        fits[i] = fit

    # # args = [(node, xs, y_true) for node in pop]
    # args = [(node, xs, y_true) for node in range(len(pop))]
    #
    # with multiprocessing.Pool(processes=4) as pool:
    #     fits = pool.starmap(fitness_helper, args)

    # print(fits)

    fits = np.nan_to_num(fits, nan=1000000, posinf=1000000)

    return fits

def correlation(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all chromosomes in a population"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T
    y_true = np.array([target_func(*list(x)) for x in xs])
    y_true_mean = np.mean(y_true)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        y_node = np.array([node(*x) for x in xs])
        y_node_mean = np.mean(y_node)
        sum_true_node = sum((y_true - y_true_mean) * (y_node - y_node_mean))
        sum_true_2 = sum((y_true - y_true_mean)**2)
        sum_node_2 = sum((y_node - y_node_mean)**2)
        R = sum_true_node / (sum_true_2 * sum_node_2) ** (1/2)
        fit = 1 - R**2
        # Post processing
        fits[i] = fit
    # Replace inf and nan to arbitrary large values
    fits = np.nan_to_num(fits, nan=1000000, posinf=1000000)
    return fits

#
# Mutation
#

def subtree_mutation(a, p_m, verbose, **kwargs):
    """Preform a mutation with a probability of p_m"""

    # Probability of mutation
    if random.random() < p_m:
        a = a.copy()
        # List of all nodes with no children
        a_nodes = [n for n in a.nodes() if len(n) == 0]
        old_brach = random.choice(a_nodes)

        if verbose > 1:
            old_a = a.copy()

        new_branch = kwargs['init_individual_func'](**kwargs)
        new_a = old_brach.replace(new_branch)

        if verbose > 1:
            print(f'Mutation: {old_a} replaces {old_brach} with {new_branch} returns {new_a}')

        a = new_a

    return a

#
# Reproduction
#

def crossover(a, b, max_subtree_depth, max_tree_depth, verbose, **kwargs):

    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()

    a_depth = a.height()
    b_depth = b.height()

    # List of all nodes with children
    a_parent_nodes = [an for an in a_new.nodes() if an.height() <= max_subtree_depth]

    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_depth = a_parent_node.height()

    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.height() <= max_subtree_depth
                      and b_depth - bn.height() + a_parent_node_depth <= max_tree_depth
                      and a_depth + bn.height() - a_parent_node_depth <= max_tree_depth
                      ]

    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)

    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    if verbose > 1:
        print(f'Crossover: {a}  &  {b}  ->  {a_new}  &  {b_new}')

    return a_new, b_new

#
# Problems
#

def logical_or(*x): return bool(x[0]) or bool(x[1])
def f(x): return x**5 - 2*x**3 + x
def mod2k(*x): return x[0] % (2 ** x[1])
def xor_and_xor(*x): return (int(x[0]) ^ int(x[1])) & (int(x[2]) ^ int(x[3]))

#
# Initial pops
#

def init_indiv(**kwargs):
    x_0 = Node('x_0')
    x_1 = Node('x_1')
    f = x_0 >> 2
    f = f.limited()
    return f

def init_sin(**kwargs):
    x = Node('x')
    f = Node.sin(x)
    f = f.limited()
    return f

#
# Default kwargs
#

kwargs = {
    'seed': None,
    'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates

    'num_reps': 1,
    'num_gens': 100,
    'pop_size': 600, #Default 600
    'max_tree_depth': 200, #Default 400
    'max_subtree_depth': 4,

    'init_individual_func': random_tree,
    'terminals': ('x',),
    'ops': ('+','-','*','/','**'),
    'p_branch': 0.5,
    'init_tree_depth': 4,

    'fitness_func': mse,
    'domains': ((0, 1, 50),),  # The domain of the problem expressed using np.linspace

    'crossover_func': crossover,
    'k': 4, # Number of randomly chosen parents for each tournament
    'p_c': 0.9, # Probability of crossover
    'keep_parents': 4, # Must be even

    'mutate_func': subtree_mutation,
    'p_m': 0.5, # Probability of mutation
}

if __name__ == '__main__':

    # kwargs['name'] = 'logical_or'
    # kwargs['target_func'] = logical_or
    # kwargs['terminals'] = ('x_0', 'x_1')
    # kwargs['domains'] = ((0,1,2), (0,1,2))
    # kwargs['num_gens'] = 10
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     ['4-ops' , ['+', '-', '*', '/']       ],
    #     ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # kwargs['name'] = 'mod'
    # kwargs['target_func'] = mod2k
    # kwargs['fitness_func'] = correlation
    # kwargs['terminals'] = ('x_0', 'x_1')
    # kwargs['domains'] = ((0,15,16), (1,2,2))
    # kwargs['init_individual_func'] = init_indiv
    # # kwargs['num_gens'] = 100
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     # ['4-ops' , ['+', '-', '*', '/']       ],
    #     ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # kwargs['name'] = 'logic'
    # kwargs['target_func'] = xor_and_xor
    # kwargs['fitness_func'] = correlation
    # kwargs['p_c'] = 0.5
    # kwargs['p_m'] = 0.5
    # kwargs['terminals'] = ('x_0', 'x_1', 'x_2', 'x_3')
    # kwargs['domains'] = ((0,1,2),(0,1,2),(0,1,2),(0,1,2))
    # kwargs['init_individual_func'] = random_tree
    # kwargs['num_gens'] = 50
    # kwargs['test_kwargs'] = [['labels','p_c','p_m']] + [[f'{p_m} {p_c}', p_c, p_m] for p_m in np.linspace(0.1,0.9,5) for p_c in np.linspace(0.1,0.9,5)]
    #
    # print(kwargs['test_kwargs'])
        # [0.3] * 2,
        # [0.5] * 2,
        # [0.7] * 2,




    kwargs['name'] = 'cos'
    kwargs['target_func'] = cos
    kwargs['fitness_func'] = correlation
    kwargs['terminals'] = ('x','e','i',)
    kwargs['domains'] = ((0, 2*math.pi, 31),)
    # kwargs['init_individual_func'] = init_sin
    kwargs['num_gens'] = 1
    kwargs['test_kwargs'] = [
        # ['labels', 'init_individual_func'],
        # ['random', random_tree],
        # ['sin', init_sin],

        ['labels', 'init_individual_func', 'fitness_func'],
        ['random', random_tree, correlation],
        ['sin', init_sin, correlation],
        ['random_mse', random_tree, mse],
        ['sin_mse', init_sin, mse],
    ]

    # kwargs['name'] = 'test'
    # kwargs['target_func'] = f
    # # kwargs['num_gens'] = 10
    # kwargs['legend_title'] = 'Types of Operations'
    # kwargs['test_kwargs'] = [
    #     ['labels', 'ops'                      ],
    #     ['4-ops' , ['+', '-', '*', '/']       ],
    #     ['5-ops' , ['+', '-', '*', '/', '**'] ],
    # ]

    # Run simulation
    all_pops, all_fits = run_sims(**kwargs)
    save_all(all_pops, all_fits, kwargs)
    plot_results(all_pops, all_fits, **kwargs)
