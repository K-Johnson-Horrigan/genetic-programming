from evolve import simulate_tests
from src.utils.plot import plot_results
from src.utils.save import load_runs, save_kwargs, load_kwargs
from genetics import *

# kwargs = {
#     'name': 'lgp_test',  # Folder to contain all results
#     'seed': None,
#     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
#     'parallelize': not True,
#     'saves_path': '../saves/',
#     # Size
#     'num_runs': 1,
#     'num_gens': 100,
#     'pop_size': 100,
#     # Initialization
#     'init_individual_func': random_code,  # Function used to generate the initial population
#     'init_min_len': 4,
#     'init_max_len': 4,
#     'max_len': 4,
#     'max_value': 4,
#     'ops': list(range(len(Linear.VALID_OPS))),
#     'addr_modes': [0,1],#list(range(len(Linear.VALID_ADDR_MODES))),
#     # Evaluation
#     'fitness_func': lgp_mse,
#     'target_func': x2,
#     'domains': [[0,4,5]],
#     # Selection
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     # Repopulation
#     'p_c': 0.0,  # Probability of crossover
#     'crossover_func': two_point_crossover,
#     'mutate_funcs': [
#         [point_mutation, 0.7],
#     ],
#     # Tests
#     'test_kwargs': [
#         ['Crossover, Mutation', 'p_c', 'mutate_funcs'],
#         *[
#             [f'{pc} {pt}', pc, [[point_mutation, pt]]]
#             # for pc in [.3,.5,.7,.9]
#             # for pt in [.3,.5,.7,.9]
#             for pc in [.9]
#             for pt in [.9]
#         ]
#     ],
# }

kwargs = {
    'name': 'self_mutate_0',  # Folder to contain all results
    'seed': None,
    'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
    'parallelize': True,
    'saves_path': '../saves/',
    # Size
    'num_runs': 10,
    'num_gens': 400,
    'pop_size': 500,
    'min_len': 3,
    'max_len': 8,
    # Initialization
    'init_individual_func': random_code,  # Function used to generate the initial population
    'init_min_len': 4,
    'init_max_len': 4,
    'max_value': 16,
    'ops': list(range(len(Linear.VALID_OPS))),
    'addr_modes': list(range(len(Linear.VALID_ADDR_MODES))),
    # Evaluation
    'fitness_func': self_mutate,
    'timeout': 64,
    # Selection
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    # Repopulation
    'p_c': 0.9,  # Probability of crossover
    'crossover_func': two_point_crossover,
    'mutate_funcs': [
        [point_mutation, 0.9],
    ],
    # Tests
    'test_kwargs': [
        ['Program Length', 'min_size', 'max_size'],
        ['Fixed', 4, 4],
        ['Changing', 3, 8],
    ],
}


if __name__ == '__main__':
    simulate_tests(**kwargs)
    pops, fits = load_runs(**kwargs)
    plot_results(pops, fits, **kwargs)