import os

import numpy as np

from utils import save_kwargs, save_run

"""Core functions that cannot be replaced"""

#
# Initialization
#

def init_pop(pop_size, init_individual_func, **kwargs):
    """Generate a random population"""
    return [init_individual_func(**kwargs) for _ in range(pop_size)]

#
# Selection
#

def tournament_selection(pop, fits, k, **kwargs):
    """Select a single parent from a tournament of k"""
    # Select the random tournament
    tourn_indices = kwargs['rng'].choice(len(pop), size=2, replace=False)
    tourn = [pop[i] for i in tourn_indices]
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(k)]
    # Sort all parents by fitness
    parent = sorted(parent)
    # Get the chromosome of the first element
    parent, fit = tourn[parent[0][1]], parent[0][0]
    return parent, fit

#
# Simulation and Iteration
#

def next_pop(pop, **kwargs):

    # Truncate Selection
    if 'lambda' in kwargs:
        # Pool starts with all current parents
        pool = list(pop) if kwargs['keep_parents'] else []
        # Create children for all parents and add to the pool
        for parent in pop:
            for i in range(kwargs['lambda']):
                child = kwargs['mutate_func'](parent, **kwargs)
                pool.append(child)
        # Evaluation
        pool_fits = kwargs['fitness_func'](pop=pool, **kwargs)
        # Sort and truncate the indices of the next generation
        pool_indices = [(pool_fits[i], i) for i in range(len(pool))]
        pool_indices = np.array(sorted(pool_indices))
        pool_indices = pool_indices[:kwargs['pop_size'], 1]
        pool_indices = list(pool_indices)
        # Reduce reps
        new_pop = np.array(pool)[pool_indices]
        kwargs['fits'] = np.array(pool_fits)[pool_indices]
        if kwargs['verbose'] > 1: print(pop)
        return new_pop, kwargs['fits']

    # Crossover
    else:
        kwargs['pop'] = pop
        # Evaluation
        kwargs['fits'] = kwargs['fitness_func'](**kwargs)
        # Elitism
        pool = [(kwargs['fits'][i], i) for i in range(kwargs['pop_size'])]
        pool = sorted(pool)
        new_pop = [pop[pool[i][1]] for i in range(kwargs['keep_parents'])]
        # Repeat until the new population is the same reps as the old
        while len(new_pop) < len(pop):
            # Selection
            c0, f0 = tournament_selection(**kwargs)
            c1, f1 = tournament_selection(**kwargs)

            # Crossover
            if kwargs['rng'].random() < kwargs['p_c']:
                c0, c1 = kwargs['crossover_func'](c0, c1, **kwargs)

            # Mutate children
            a, p = zip(*kwargs['mutate_funcs'])
            mutate_func = kwargs['rng'].choice(a=a, p=p)
            if mutate_func is not None:
                c0 = mutate_func(c0, **kwargs)
                c1 = mutate_func(c1, **kwargs)

            c0 = c0.copy()
            c1 = c1.copy()
            c0.prev_fit = (f0 + f1) / 2
            c1.prev_fit = (f0 + f1) / 2
            new_pop.append(c0)
            new_pop.append(c1)

        return new_pop, kwargs['fits']


def simulate_run(**kwargs):
    """Run a single simulation of a full set of generations"""

    # Add no operation as a possible mutation
    prob_noop = 1 - sum(list(zip(*kwargs['mutate_funcs']))[1])
    if prob_noop > 0:
        kwargs['mutate_funcs'].append([None, prob_noop])

    shape = (kwargs['num_gens'] + 1, kwargs['pop_size'])

    # Initialization

    # all_pops = [pop]

    all_pops = np.empty(shape, dtype=[('verts','object'),('edges','object')])
    all_fits = np.empty(shape)
    # all_verts = []
    # all_edges = []
    prev_fit = np.empty(shape)

    pop = init_pop(**kwargs)
    all_pops[0] = [n.to_lists() for n in pop]

    # for indiv in pop: indiv.prev_fit = 0

    # Loop level 2
    for generation in range(kwargs['num_gens']):
        if kwargs['verbose'] > 0:
            print(f'\tGeneration {generation} of {kwargs["num_gens"]}')

        # Next generation and previous fitness
        pop, fit = next_pop(pop=pop, **kwargs)

        # Save results
        all_pops[generation + 1] = [n.to_lists() for n in pop]
        all_fits[generation] = fit

        # verts, edges = list(zip(*[n.to_lists() for n in pop]))
        # all_verts.append(verts)
        # all_edges.append(edges)

    # Final fitness values
    # all_fits.append(kwargs['fitness_func'](pop, is_final=True, **kwargs))
    all_fits[-1] = kwargs['fitness_func'](pop, is_final=True, **kwargs)

    return all_pops, all_fits


def simulate_tests(num_runs, test_kwargs, **kwargs):
    """
    Simulate all runs for all tests with different hyperparameters.
    There are four levels: [test] [run/replicant] [generation/population] [individual]
    """

    # Save kwargs first in case of failure
    save_kwargs(num_runs=num_runs, test_kwargs=test_kwargs, **kwargs)

    num_tests = len(test_kwargs) - 1

    # This can be saved as a 4D array for easy manipulation and access
    # [test] [replicant] [generation] [individual]
    # all_pops = np.empty((num_tests, num_runs, kwargs['num_gens']+1, kwargs['pop_size']), dtype=object)
    # all_fits = np.empty((num_tests, num_runs, kwargs['num_gens']+1, kwargs['pop_size']))

    # Loop level 0
    for test_num in range(num_tests):

        # Each test is saved in its own directory
        path = 'saves/' + kwargs['name'] + '/data/' + test_kwargs[test_num + 1][0]
        os.makedirs(path, exist_ok=True)

        if kwargs['verbose'] > 0: print(f'Test {test_num}')
        # Modify kwargs
        for key, value in zip(test_kwargs[0], test_kwargs[test_num + 1]):
            if kwargs['verbose'] > 0: print(f'{key}: {value}')
            kwargs[key] = value

        # Loop level 1
        for run in range(num_runs):

            # Set random seed
            # TODO: Implement properly for parallelization
            if kwargs['seed'] is None:
                kwargs['seed'] = np.random.randint(0, 2 ** 64, dtype='uint64')
            kwargs['rng'] = np.random.default_rng(kwargs['seed'])

            pops, fits = simulate_run(**kwargs)

            save_run(path, pops, fits, **kwargs)
            # all_pops[test_num, run] = pops
            # all_fits[test_num, run] = fits

    # return all_pops, all_fits