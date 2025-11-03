from src.evolve import simulate_tests
from src.models.network import *
from src.utils.plot import plot_results
from src.utils.save import load_fits


kwargs = {
    'name': 'test_0',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': True,
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
    'i_c': [2, 1.125, 0.75, 0.375, 0.125, 0],  # 2M band
    #'i_c': [2, 0.625, 0.375, 0.125, 0],  # 5.5M band
    #'i_c': [2, 0.5, 0.375, 0.125, 0],  # 11M band
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
        ['All', list(range(11))],
        ['Orthogonal', (1,6,11)],
        ['1-6', (1,2,3,4,5,6)],
    ],
}


if __name__ == '__main__':
    # Setup the problem TODO improve implementation of procedural problem setup
    nodes, links = regular_topology((3,3))
    kwargs = setup(nodes, links, **kwargs)
    # Run evolution
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)