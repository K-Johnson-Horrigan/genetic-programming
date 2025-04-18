import multiprocessing

import numpy as np
from scipy.optimize import minimize

from node import *
from evolve import *
from plot import *
from utils import save_all

from math import sin, cos

"""Functions relevant to implementing genetic programming"""


#
# Initialization Functions
#

def random_tree(init_tree_depth, ops=Node.valid_ops, terminals=('x',), p_branch=0.5, init_call=True, **kwargs):
    """Generate a random tree"""
    # Create a branch with an operator value
    if init_call or random.random() < p_branch and init_tree_depth > 0:
        op = random.choice(ops)
        children = [random_tree(init_tree_depth - 1, ops, terminals, p_branch, False) for _ in range(Node.valid_ops[op])]
        return Node(op, children)
    # Create a leaf
    else:
        return Node(random.choice(terminals))


def random_noop_tree(init_tree_depth, num_registers, ops=Node.valid_ops, terminals=('x',), p_branch=0.5, **kwargs):
    c = [
        random_tree(init_tree_depth-1, ops=ops, terminals=terminals, p_branch=p_branch, init_call=True, **kwargs)
        for _ in range(num_registers)
    ]
    return Node('noop', c)


#
# Fitness Functions
#

def fitness_helper(id, node, xs, y_target):
    """Used for parallel computing"""
    y_actual = [node(*x) for x in xs]
    fit = (sum((abs(y_target - y_actual)) ** 2) / len(xs)) ** (1 / 2)
    return fit


def mse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1))
    y_target = np.array([target_func(*list(x)) for x in xs.T])
    # xs = xs.swapaxes(0, 1)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        # Pass all test cases as a single numpy array so that a semantic vector can be formed if needed
        y_actual = node(*xs, eval_method=kwargs['eval_method'])
        fit = (sum((abs(y_target - y_actual)) ** 2) / len(xs)) ** (1/2)
        fits[i] = fit
    # args = [(id, node, xs, y_target) for id,node in enumerate(pop)]
    # with multiprocessing.Pool(processes=4) as pool:
    #     fits = pool.starmap(fitness_helper, args)
    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits


def correlation(pop, target_func, domains, is_final=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1))
    y_target = np.array([target_func(*list(x)) for x in xs.T])
    y_target_mean = np.mean(y_target)
    fits = np.empty(len(pop))
    for i,node in enumerate(pop):
        # Pass all test cases as a single numpy array so that a semantic vector can be formed if needed
        y_actual = node(*xs, eval_method=kwargs['eval_method'])
        y_actual_mean = np.mean(y_actual)
        sum_target_actual = sum((y_target - y_target_mean) * np.conjugate(y_actual - y_actual_mean))
        sum_target_2 = sum(abs((y_target - y_target_mean))**2)
        sum_actual_2 = sum(abs((y_actual - y_actual_mean))**2)
        R = sum_target_actual / (sum_target_2 * sum_actual_2) ** (1/2)
        fit = 1 - R * np.conjugate(R)
        fits[i] = fit

        # Post-processing
        if is_final:
            def min_f(a): return np.sum(np.abs(y_target - (a[1] * y_actual + a[0])))
            res = minimize(min_f, [0,0], method='Nelder-Mead', tol=1e-6)
            new_node = (node * float(res.x[1])) + float(res.x[0])
            pop[i] = new_node
    # Replace inf and nan to arbitrary large values
    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits


#
# Mutation Functions
#

def randomize_mutation(root, **kwargs):
    """Return a random new individual"""
    return kwargs['init_individual_func'](**kwargs)


def subtree_mutation(root, **kwargs):
    """Swap a random node with a random new subtree"""
    new_root = root.copy()
    new_branch = kwargs['new_individual_func'](**kwargs)
    new_branch_height = new_branch.height()
    # List of all nodes with no children
    root_nodes = [
        n for n in new_root.nodes() if len(n) == 0 and n.depth() + new_branch_height < kwargs['max_tree_depth']
    ]
    # Failure occurs if there is no way to insert the new tree
    if len(root_nodes) == 0:
        if kwargs['verbose'] > 1:
            print(f'\tsubtree_mutation: failed for {new_root} and branch {new_branch} of height {new_branch_height}')
        return new_root
    # Select and replace a node with the branch
    branch = random.choice(root_nodes)
    branch.replace(new_branch)
    if kwargs['verbose'] > 1:
        print(f'\tsubtree_mutation: {root} replaces {branch} with {new_branch} returns {new_root}')
    return new_root


def pointer_mutation(root, **kwargs):
    """Randomly change an op node to point to a random node that is not an ancestor"""
    new_root = root.copy()
    # List of all nodes with parents and children
    # The root node and terminal nodes cannot have their pointer changed
    valid_parent_nodes = [n for n in new_root.nodes() if len(n.parents) > 0 and len(n.children) > 0]
    # Pointer mutations will fail in situations such as if the graph is a path
    if len(valid_parent_nodes) == 0:
        if kwargs['verbose'] > 1:
            print(f'\tpointer_mutation: failed for {new_root}')
        return new_root
    # Select a random parent to have its pointer changed
    parent_node = random.choice(valid_parent_nodes)
    child_node_index = np.random.randint(len(parent_node))
    old_child_node = parent_node[child_node_index]
    # Select a new child
    new_child_node = random.choice([n for n in new_root.nodes() if parent_node.index_in(n.nodes()) == -1])
    # Replace the child
    parent_node[child_node_index] = new_child_node
    # Recalculate parents
    new_root.reset_parents()
    new_root.set_parents()

    if kwargs['verbose'] > 1:
        print(f'\tpointer_mutation: {root} replaces {old_child_node} with {new_child_node} returns {new_root}')
    return new_root


def split_mutation(root, **kwargs):
    """Only the direct parent may """
    new_root = root.copy()
    # List of all nodes with multiple parents
    valid_child_nodes = [n for n in new_root.nodes() if len(n.parents) > 1]
    # Mutation failed
    if len(valid_child_nodes) == 0:
        print(f'\tsplit_mutation: failed for {new_root}')
        return new_root
    child_node = random.choice(valid_child_nodes)
    # Shallow copy the node for each parent
    for parent in child_node.parents:
        parent[child_node.index_in(parent)] = Node(child_node.value, child_node.children)
    # Recalculate parents
    new_root.reset_parents()
    new_root.set_parents()
    if kwargs['verbose'] > 1:
        print(f'\tsplit_mutation: {root} splits {child_node} returns {new_root}')
    return new_root


def deep_split_mutation(root, **kwargs):
    """Only the direct parent may """
    new_root = root.copy()
    # List of all nodes with multiple parents
    valid_child_nodes = [n for n in new_root.nodes() if len(n.parents) > 1]
    # Mutation failed
    if len(valid_child_nodes) == 0:
        print(f'\tsplit_mutation: failed for {new_root}')
        return new_root
    child_node = random.choice(valid_child_nodes)
    # Deep copy the node for each parent
    for parent_node in child_node.parents:
        parent_node[child_node.index_in(parent_node)] = child_node.copy()
    # Recalculate parents
    new_root.reset_parents()
    new_root.set_parents()
    if kwargs['verbose'] > 1:
        print(f'\tsplit_mutation: {root} splits {child_node} returns {new_root}')
    return new_root


#
# Crossover Functions
#

def subtree_crossover(a, b, **kwargs):
    # Copy original trees
    a_new = a.copy()
    b_new = b.copy()
    a_height = a.height()
    b_height = b.height()
    # List of all nodes
    a_parent_nodes = [an for an in a_new.nodes() if an.height() <= kwargs['max_subtree_depth']]
    # Select the first random node (branch)
    a_parent_node = random.choice(a_parent_nodes)
    a_parent_node_height = a_parent_node.height()
    # List of all nodes that could swap with a without being too long in the worse case
    # TODO implement a more accurate assessment of length
    b_parent_nodes = [bn for bn in b_new.nodes() if bn.height() <= kwargs['max_subtree_depth']
                      and b_height - bn.height() + a_parent_node_height <= kwargs['max_tree_depth']
                      and a_height + bn.height() - a_parent_node_height <= kwargs['max_tree_depth']
                      ]

    if len(b_parent_nodes) == 0:
        print(f'\tsubtree_crossover: failed between {a} and {b}')
        return a, b

    # Select a random node with children
    b_parent_node = random.choice(b_parent_nodes)

    # Swap the two nodes
    a_parent_node.replace(b_parent_node.copy())
    b_parent_node.replace(a_parent_node.copy())

    if kwargs['verbose'] > 1:
        print(f'\tsubtree_crossover: {a} and {b} produce {a_new} and {b_new}')
    return a_new, b_new


#
# Target Functions
#

def logical_or(*x): return bool(x[0]) or bool(x[1])
def f(x): return x**5 - 2*x**3 + x
def mod2k(*x): return x[0] % (2 ** x[1])
def xor_and_xor(*x): return (int(x[0]) ^ int(x[1])) & (int(x[2]) ^ int(x[3]))
# def const_32(x): return x**5 + 32*x**3 + x
def const_32(x): return 32*x**2 + x


#
# Initial pops
#

def init_indiv(**kwargs):
    x_0 = Node('x_0')
    x_1 = Node('x_1')
    f = x_0 >> 2
    f = f.limited()
    return f

def init_sin(**kwargs): return Node.sin(Node('x'))
def init_sin_limited(**kwargs): return Node.sin(Node('x')).limited()

#
# Debug
#

if __name__ == '__main__':

    # pass
    # f0 = x + 1
    # f1 = f0 + x
    # f2 = f1 + f0
    # f = f2
    #
    g = x + 1
    f = g + g

    # f = random_noop_tree(3,5, ['+','-','*','/','**'],['x'],1)

    # plot_graph(f)

    f = subtree_mutation(
        f,
        verbose=2,
        max_tree_depth=5,
        new_individual_func=random_tree,
        init_tree_depth=2,
        ops=['+','-','*','/','**']
    )

    plot_graph(f)

    # f = pointer_mutation(f, verbose=2)
    # f =
    # plot_graph(split_mutation(f, verbose=2))
    # plot_graph(deep_split_mutation(f, verbose=2))

    # f = e ** (i * x)
    # # f = Node.sin(x)
    # # f = 3j * (x * i + 3j)
    #
    # f1 = Node.cos(x)
    # f2 = i * Node.sin(x)
    #
    # def ff(x):
    #     return cos(x)
    #     # return x * 1j
    #
    # pop = [f, f1, f2]
    #
    # # fits = correlation(pop, ff, domains=[[0, 2*np.pi, 1]])
    #
    # plot_nodes(pop,
    #            labels=['$e^{ix}$', '$\\cos(x)$', '$\\sin(x)$'],
    #            target_func=ff,
    #            result_fitness_func=mse, #correlation,
    #            domains=[[0, 2*np.pi, 31]])



    # x = Node('x')
    #
    # # f1 = 1 + x
    # # f2 = x - f1
    # # f3 = f1 * f2
    # # f4 = f2 / f3
    # # f = f4
    #
    # # f1 = 1 + x
    # # f2 = f1 - x
    # # f3 = f2 * f1
    # # f4 = f3 / f2
    # # f = f4
    #
    # f = (x**5 - 2*x**3 + x).to_tree()
    #
    # # g1 = x / 1
    #
    # # (f1).replace(g1)
    #
    # plot_graph(f)
    # f = pointer_mutation(f)
    # plot_graph(f)


