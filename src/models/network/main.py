import numpy as np

from src.evolve import simulate_tests
from src.models.network.methods import *
from src.utils.plot import plot_results
from src.utils.save import load_runs, load_fits


kwargs = {
    'name': 'test_0',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': not True,
    'saves_path': '../../../saves/network/',  # Save path relative to this file
    ## Size ##
    'num_runs': 1,
    'num_gens': 100,
    'pop_size': 100,
    ## Initialization ##
    'channels': list(range(1,12)),
    'init_individual_func': random_network,  # Function used to generate a new organism
    ## Evaluation ##
    'fitness_func': total_interference,
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'crossover_funcs': [
        [two_point_crossover, 0.9],
    ],
    'mutate_funcs': [
        [point_mutation, 0.0],
    ],
    ## Tests ##
    'test_kwargs': [
        ['Channels', 'channels'],
        ['11', list(range(11))],
        ['ortho', (1,6,11)],
        ['test', (1,2)],
    ],
}

if __name__ == '__main__':

    # kwargs['rng'] = np.random.default_rng(2)

    nodes, links = regular_topology()
    kwargs = setup(nodes, links, **kwargs)
    # kwargs['min_len'] = len(kwargs['links'])
    # kwargs['max_len'] = len(kwargs['links'])

    # org = random_network(**kwargs)
    #
    # h = fitness(org, **kwargs)
    #
    # pass

    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)