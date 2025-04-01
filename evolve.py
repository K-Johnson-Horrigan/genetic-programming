import random
import numpy as np

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
    tourn = random.choices(pop, k=k)
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
        # Reduce size
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
        # Repeat until the new population is the same size as the old
        while len(new_pop) < len(pop):
            # Selection
            c0, f0 = tournament_selection(**kwargs)
            c1, f1 = tournament_selection(**kwargs)
            # prev_fit = np.mean(kwargs['fitness_func'](pop=[c0,c1], **kwargs))

            # Crossover
            if random.random() < kwargs['p_c']:
                c0, c1 = kwargs['crossover_func'](c0, c1, **kwargs)
                # cc0.prev_fit

            # Mutate children
            a, p = zip(*kwargs['mutate_funcs'])
            mutate_func = np.random.choice(a=a, p=p)

            if mutate_func is not None:
                if kwargs['verbose'] > 1:
                    print(mutate_func.__name__)
                c0 = mutate_func(c0, **kwargs)
                c1 = mutate_func(c1, **kwargs)

            new_pop.append(c0)
            new_pop.append(c1)
        return new_pop, kwargs['fits']


def run_sim(**kwargs):
    """Run a single simulation of a full set of generations"""

    # kwargs['mutate_funcs'] = np.array(kwargs['mutate_funcs'])

    # Add no operation as a possible mutation
    noop = 1 - sum(list(zip(*kwargs['mutate_funcs']))[1])
    kwargs['mutate_funcs'].append([None, noop])

    # Set random seed
    if 'seed' in kwargs:
        random.seed(kwargs['seed'])
        np.random.seed(kwargs['seed'])

    # Initial
    pop = init_pop(**kwargs)
    all_pops = [pop]
    all_fits = []

    for generation in range(kwargs['num_gens']):
        if kwargs['verbose'] > 0:
            print(f'Generation {generation} of {kwargs["num_gens"]}')

        # Next generation and previous fitness
        pop, fit = next_pop(pop=pop, **kwargs)
        all_fits.append(fit)
        all_pops.append(pop)

    # Final fitness values
    all_fits.append(kwargs['final_fitness_func'](all_pops[-1], **kwargs))

    return all_pops, all_fits


def run_sims(num_reps, test_kwargs, **kwargs):
    """Run multiple tests with different hyperparameters"""

    num_tests = len(test_kwargs) - 1

    # This can be saved as a 4D array for easy manipulation and access
    # [test] [replicant] [generation] [individual]
    all_pops = np.empty((num_tests, num_reps, kwargs['num_gens'] + 1, kwargs['pop_size']), dtype=object)
    all_fits = np.empty((num_tests, num_reps, kwargs['num_gens'] + 1, kwargs['pop_size']))

    # Tests
    for test_num in range(num_tests):
        # changes = test_kwargs[test_num]
        if kwargs['verbose'] > 0: print(f'Test {test_num}')
        for key,value in zip(test_kwargs[0], test_kwargs[test_num + 1]):
            if kwargs['verbose'] > 0: print(f'\t{key}: {value}')
            kwargs[key] = value
        for rep in range(num_reps):
            pops, fits = run_sim(**kwargs)
            all_pops[test_num, rep] = pops
            all_fits[test_num, rep] = fits

    return all_pops, all_fits
