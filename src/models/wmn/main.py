import os
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_directory[:-14])

from src.evolve import simulate_tests
from src.models.wmn import *
from src.models.wmn.plot import plot_results, plot_extra_1
from src.utils.save import load_fits, load_extra_1

kwargs = {
    'name': 'example_8',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': False,
    'saves_path': 'saves/',  # Save path relative to this file
    ## Size ##
    'num_runs': 2,
    'num_gens': 50,
    'pop_size': 16,
    'num_routers': 16,
    'num_clients': 48,
    'min_value':  0,  # Min value of a position (inclusive)
    'max_value': 16,  # Max value of a position (exclusive)
    'radius': 2,  # Coverage radius of each router
    ## Setup ##
    'setup_func': setup_uniform_clients,  # Function that runs for each unique job and returns updated kwargs
    ## Initialization ##
    'init_individual_func': random_uniform_router_coords,  # Function used to generate a new organism
    ## Evaluation ##
    'fitness_func': lexicase_fitness,
    'epsilon': 0,
    ## Selection ##
    'minimize_fitness': False,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    ## Extra data collection ##
    'extra_data': True,
    'extra_data_1': get_con_vals,
    'extra_data_2': get_cov_vals,

    ## Repopulation ##
    'crossover_funcs': [
        [coords_two_point_crossover, 0.8],
    ],
    'mutate_funcs': [
        [coords_point_mutation_2d, 0.2],
    ],
    ## Tests ##
    # 'test_kwargs': [
    #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
    #     # ['0.0 0.0', [[two_point_crossover, 0.0]], [[point_mutation, 0.0]]],
    #     *[
    #         [f'{c}, {m}', [[coords_two_point_crossover, m]], [[coords_point_mutation_2d, m]]]
    #         for c in [.2]
    #         for m in [.2]
    #     ]
    # ]
    'test_kwargs': [
        ['Crossover Function', 'crossover_funcs'],
        ['Two Point', [[coords_two_point_crossover, 0.8]]],
        ['Fixed', [[coords_fixed_two_point_crossover, 0.8]]],
        ['Reordered', [[coords_reorder_crossover, 0.8]]],
    ]
}

# kwargs = {
#     'name': 'fitness_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../../../saves/placement/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 15,
#     'num_gens': 50,
#     'pop_size': 16,
#     'num_routers': 16,
#     'num_clients': 48,
#     'min_value':  0,  # Min value of a position (inclusive)
#     'max_value': 16,  # Max value of a position (exclusive)
#     'radius': 2,  # Coverage radius of each router
#     ## Setup ##
#     'setup_func': setup_uniform_clients,  # Function that runs for each unique job and returns updated kwargs
#     ## Initialization ##
#     'init_individual_func': random_uniform_router_coords,  # Function used to generate a new organism
#     ## Evaluation ##
#     'fitness_func': cov_con_entropy_fitness,
#     ## Selection ##
#     'minimize_fitness': False,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         [coords_two_point_crossover, 0.8],
#     ],
#     'mutate_funcs': [
#         [coords_point_mutation_2d, 0.2],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Fitness Function', 'fitness_func'],
#         ['Sum', cov_con_sum_fitness],
#         ['Entropy', cov_con_entropy_fitness],
#     ]
# }

if __name__ == '__main__':
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)
