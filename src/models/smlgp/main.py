from src.evolve import simulate_tests
from src.models.smlgp import *
from src.utils.plot import plot_results
from src.utils.save import load_runs, load_fits

# kwargs = {
#     'name': 'self_rep_mult_0',  # Name of folder to contain all results
#     'seed': None,
#     'verbose': True,
#     'parallelize': True,
#     'saves_path': '../saves/',  # Save path relative to this file
#     ## Size ##
#     'num_runs': 8,
#     'num_gens': 1000,
#     'pop_size': 1000,
#     'min_len': 4,
#     'max_len': 4,
#     ## Initialization ##
#     'init_individual_func': random_self_rep_code,  # Function used to generate a new organism
#     'init_min_len': 4,
#     'init_max_len': 4,
#     'max_value': 16,
#     'ops': list(range(len(Linear.DEFAULT_OPS))),
#     'addr_modes': list(range(len(Linear.DEFAULT_ADDR_MODES))),
#     ## Evaluation ##
#     'fitness_func': lgp_self_rep_rmse,
#     'target_func': multiply,  # The function that the organism is attempting to replicate
#     'domains': [list(range(0, 4)), list(range(0, 4))],  # Cases are generated from the Cartesian product
#     'timeout': 64,  # Number of evaluation iterations before forced termination
#     ## Selection ##
#     'minimize_fitness': True,
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     ## Repopulation ##
#     'crossover_funcs': [
#         # [two_point_crossover, 0.9],
#         [self_crossover, 1.0],
#     ],
#     'mutate_funcs': [
#         [point_mutation, 0.0],
#     ],
#     ## Tests ##
#     'test_kwargs': [
#         ['Max Length', 'init_max_len', 'max_len'],
#         *[[str(i),i,i] for i in [4,8,16]],
#     ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     *[
#     #         [f'{pc} {pm}', [[two_point_crossover, pc]], [[point_mutation, pm]]]
#     #         for pc in [.3,.5,.7,.9]
#     #         for pm in [.3,.5,.7,.9]
#     #         # for pc in [.9]
#     #         # for pt in [.9]
#     #     ]
#     # ],
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
#     #     [f'0.5', [[two_point_crossover, .5]], [[point_mutation, .5]]],
#     #     [f'0.9', [[two_point_crossover, .9]], [[point_mutation, .9]]],
#     # ],
# }


kwargs = {
    'name': 'new_test_0',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': True,
    'saves_path': '../../../saves/smlgp/',  # Save path relative to this file
    ## Size ##
    'num_runs': 12,
    'num_gens': 100,
    'pop_size': 100,
    'min_lens': [64, 64],  # The length of each memory
    'max_lens': [64, 64],  # The length of each memory
    # 'min_len': 4,  # Used by predefined repopulation methods
    # 'max_len': 4,  # Used by predefined repopulation methods
    ## Initialization ##
    'init_individual_func': random_code,  # Function used to generate a new organism
    # 'init_min_len': 4,  # Used by predefined initialization methods
    # 'init_max_len': 4,  # Used by predefined initialization methods
    'init_min_lens': [64, 64],  # The length of each memory
    'init_max_lens': [64, 64],  # The length of each memory
    'max_value': 16,
    'ops': list(range(len(Linear.DEFAULT_OPS))),
    'addr_modes': list(range(len(Linear.DEFAULT_ADDR_MODES))),
    ## Evaluation ##
    'fitness_func': lgp_rmse,
    'target_func': multiply,  # The function that the organism is attempting to replicate across the domains
    'domains': [list(range(0, 4)), list(range(0, 4))],  # Cases are generated from the Cartesian product
    'timeout': 64,  # Number of evaluation iterations before forced termination
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'crossover_funcs': [
        [two_point_crossover, 0.9],
        # [self_crossover, 1.0],
    ],
    'mutate_funcs': [
        [point_mutation, 0.0],
    ],
    ## Tests ##
    # 'test_kwargs': [
    #     ['Crossover Rate', 'init_individual_func', 'fitness_func', 'crossover_funcs', ],
    #     # ['1.0', random_non_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
    #     # ['0.7', random_random_code, lgp_self_rep_rmse, [[self_crossover, 0.7]]],
    #     # ['1.0', random_random_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
    #     # ['0.5', random_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 0.5]]],
    #     # ['1.0', random_self_rep_code, lgp_self_rep_rmse, [[self_crossover, 1.0]]],
    #     ['p.5', random_code, lgp_rmse, [[two_point_crossover, 1.0]], ('STOP','LOAD','STORE','ADD','SUB','IFEQ','RAND',)],
    #     # ['p.5', random_code, lgp_rmse, [[two_point_crossover, 1.0]]],
    # ],
    'test_kwargs': [
        ['Ops', 'ops'],
        ['Normal', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',)],
        ['DEL', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'DEL',)],
    ],
    # 'test_kwargs': [
    #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
    #     *[
    #         [f'{pc} {pm}', [[two_point_crossover, pc]], [[point_mutation, pm]]]
    #         for pc in [.3,.5,.7,.9]
    #         for pm in [.3,.5,.7,.9]
    #         # for pc in [.9]
    #         # for pt in [.9]
    #     ]
    # ],
    # 'test_kwargs': [
    #     ['Crossover, Mutation', 'crossover_funcs', 'mutate_funcs'],
    #     [f'0.5', [[two_point_crossover, .5]], [[point_mutation, .5]]],
    #     [f'0.9', [[two_point_crossover, .9]], [[point_mutation, .9]]],
    # ],
}

if __name__ == '__main__':
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)



