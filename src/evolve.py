"""
Core functions used in controlling evolution
All functions are independent of the subject of evolution
"""
import copy
from multiprocessing import Pool, cpu_count

import numpy as np

from src.utils.save import save_kwargs, save_run, save_run_verbose

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
    tourn_indices = kwargs['rng'].choice(len(pop), size=k, replace=False)
    tourn = [pop[i] for i in tourn_indices]
    # Created a zipped list of fitness and chromosomes
    parent = [(fits[i], i) for i in range(k)]
    # Sort all parents by fitness
    parent = sorted(parent)
    if not kwargs['minimize_fitness']:
        parent = parent[::-1]
    # Get the chromosome of the first element
    parent, fit = tourn[parent[0][1]], parent[0][0]
    return parent, fit


#
# Simulation and Iteration
#

def next_pop(pop, gen, **kwargs):
    """Returns the fitness values for the given population and the next population"""

    # Truncate Selection
    # This is not used
    if 'lambda' in kwargs:
        pass
        # Pool starts with all current parents
        # pool = list(pop) if kwargs['keep_parents'] else []
        # # Create children for all parents and add to the pool
        # for parent in pop:
        #     for i in range(kwargs['lambda']):
        #         child = kwargs['mutate_func'](parent, **kwargs)
        #         pool.append(child)
        # # Evaluation
        # pool_fits = kwargs['fitness_func'](pop=pool, **kwargs)
        # # Sort and truncate the indices of the next generation
        # pool_indices = [(pool_fits[i], i) for i in range(len(pool))]
        # pool_indices = np.array(sorted(pool_indices))
        # pool_indices = pool_indices[:kwargs['pop_size'], 1]
        # pool_indices = list(pool_indices)
        # # Reduce reps
        # new_pop = np.array(pool)[pool_indices]
        # kwargs['fits'] = np.array(pool_fits)[pool_indices]
        # if kwargs['verbose'] > 1: print(pop)
        # return new_pop, kwargs['fits']

    # Crossover
    else:
        kwargs['pop'] = pop

        # Evaluation
        kwargs['fits'] = kwargs['fitness_func'](gen=gen, **kwargs)

        # Elitism
        pool = [(kwargs['fits'][i], i) for i in range(kwargs['pop_size'])]
        pool = sorted(pool)
        pool = pool if kwargs['minimize_fitness'] else pool[::-1]
        new_pop = [pop[pool[i][1]] for i in range(kwargs['keep_parents'])]

        # Repeat until the new population is the same reps as the old
        while len(new_pop) < len(pop):

            # Selection
            org_0, fit_0 = tournament_selection(**kwargs)
            org_1, fit_1 = tournament_selection(**kwargs)
            org_0 = copy.deepcopy(org_0)
            org_1 = copy.deepcopy(org_1)

            # Crossover
            a, p = zip(*kwargs['crossover_funcs'])
            crossover_func = kwargs['rng'].choice(a=a, p=p)
            if crossover_func is not None:
                org_0, org_1 = crossover_func(org_0, org_1, gen=gen, **kwargs)

            # Mutation
            a, p = zip(*kwargs['mutate_funcs'])
            mutate_func = kwargs['rng'].choice(a=a, p=p)
            if mutate_func is not None:
                org_0 = mutate_func(org_0, gen=gen, **kwargs)
                org_1 = mutate_func(org_1, gen=gen, **kwargs)

            new_pop.append(org_0)
            new_pop.append(org_1)

        return kwargs['fits'], new_pop


def simulate_run(**kwargs):
    """Run a single simulation of a full set of generations"""

    # Initialization
    shape = (kwargs['num_gens'] + 1, kwargs['pop_size'])
    run_pops = np.empty(shape, dtype=object)
    run_fits = np.empty(shape)

    run_extra_data_1 = np.empty(shape)
    run_extra_data_2 = np.empty(shape)

    pop = init_pop(**kwargs)
    run_pops[0] = pop

    # Repeat for each generation
    for generation in range(kwargs['num_gens']):

        if kwargs['verbose'] > 0 and generation % 1 == 0:
            print(f'Simulating Test {kwargs["test_name"]}, Run {kwargs["seed"]}, Generation {generation} of {kwargs["num_gens"]}')

        # Remove node
        # if generation == kwargs.get('nodes_removed_gen'):
        #     for node_index in kwargs['nodes_removed']:
        #         print(f'Removing node {node_index}')
        #         kwargs['interf_mask'] = remove_node(node_index, **kwargs)

        # Next generation and previous fitness
        fit, pop = next_pop(pop=pop, gen=generation, **kwargs)

        # Save results
        # all_pops[generation + 1] = [n.to_lists() for n in pop]
        run_pops[generation + 1] = pop
        run_fits[generation] = fit
        if(kwargs['extra_data']):
            run_extra_data_1[generation] = kwargs['extra_data_1'](pop, **kwargs)
            run_extra_data_2[generation] = kwargs['extra_data_2'](pop, **kwargs)

    # Final fitness values
    run_fits[-1] = kwargs['fitness_func'](pop, gen=generation, is_final=True, **kwargs)
    if(kwargs['extra_data']):
            run_extra_data_1[-1] = kwargs['extra_data_1'](pop, **kwargs)
            run_extra_data_2[-1] = kwargs['extra_data_2'](pop, **kwargs)

    return run_pops, run_fits, run_extra_data_1, run_extra_data_2


def _simulate_and_save_test_run(test_num, run_num, test_kwargs, base_kwargs):
    """Parallel worker for (test_num, run_num)."""

    # Extract test-specific kwargs
    test_keys = test_kwargs[0]
    test_values = test_kwargs[test_num + 1]
    kwargs = base_kwargs.copy()

    # Update with test-specific values
    for key, value in zip(test_keys, test_values):
        kwargs[key] = value


    # Set path and create directory (thread-safe)
    test_name = test_values[0]
    test_path = f'{kwargs["saves_path"]}{kwargs["name"]}/data/{test_values[0]}'

    # Assign seed and RNG
    if kwargs['seed'] is None:
        kwargs['seed'] = np.random.randint(0, 2**64, dtype='uint64')
    kwargs['rng'] = np.random.default_rng(kwargs['seed'])

    if 'setup_func' in kwargs:
        kwargs = kwargs['setup_func'](**kwargs)

    # Add no-operation as a possible crossover
    prob_noop = 1 - sum(list(zip(*kwargs['crossover_funcs']))[1])
    if prob_noop > 0:
        kwargs['crossover_funcs'].append([None, prob_noop])

    # Add no-operation as a possible mutation
    prob_noop = 1 - sum(list(zip(*kwargs['mutate_funcs']))[1])
    if prob_noop > 0:
        kwargs['mutate_funcs'].append([None, prob_noop])

    # Run simulation and save
    pops, fits, extra_1, extra_2 = simulate_run(test_name=test_name, **kwargs)
    if(kwargs['extra_data']): save_run_verbose(test_path, pops, fits, extra_1, extra_2, **kwargs)
    else: save_run(test_path, pops, fits, **kwargs)


def simulate_tests(num_runs, test_kwargs, **kwargs):
    """
    Simulate all runs for all tests with different hyperparameters.
    There are four levels: [test] [run/replicant] [generation/population] [organism/individual]
    """

    # Save kwargs first in case of failure
    save_kwargs(num_runs=num_runs, test_kwargs=test_kwargs, **kwargs)

    # Number of tests must be inferred and is only used within this function
    num_tests = len(test_kwargs) - 1

    # Build the job list: one job per (test, run)
    jobs_args = [
        (test_num, run_num, test_kwargs, kwargs.copy())
        for test_num in range(num_tests)
        for run_num in range(num_runs)
    ]

    # Parallelize processes
    if kwargs.get('parallelize', False):

        if kwargs.get('verbose', 0) > 0:
            print(f'Dispatching {len(jobs_args)} parallel jobs')

        #MODIFY cpu_count() to adjust max core count use
        with Pool(processes=min(cpu_count(), len(jobs_args))) as pool:
            results = pool.starmap(_simulate_and_save_test_run, jobs_args)

    # Non parallelized
    else:
        for job_args in jobs_args:
            _simulate_and_save_test_run(*job_args)