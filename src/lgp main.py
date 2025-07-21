from evolve import simulate_tests
from genetics import *
from src.utils.plot import plot_results
from src.utils.save import load_runs


kwargs = {
    'name': 'mult_2',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': True,
    'saves_path': '../saves/',  # Save path relative to this file
    ## Size ##
    'num_runs': 5,
    'num_gens': 200,
    'pop_size': 200,
    'min_len': 4,
    'max_len': 4,
    ## Initialization ##
    'init_individual_func': random_code,  # Function used to generate the initial population
    'init_min_len': 4,
    'init_max_len': 4,
    'max_value': 16,
    'ops': list(range(len(Linear.VALID_OPS))),
    'addr_modes': list(range(len(Linear.VALID_ADDR_MODES))),
    ## Evaluation ##
    'fitness_func': lgp_error,
    'target_func': multiply,
    'domains': [list(range(5)), list(range(5))],  # Cases are generated from the Cartesian product
    'timeout': 64,
    ## Selection ##
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    ## Repopulation ##
    'crossover_funcs': [
        [two_point_crossover, 0.9],
    ],
    'mutate_funcs': [
        [point_mutation, 0.9],
    ],
    ## Tests ##
    'test_kwargs': [
        ['Max Length', 'init_max_len', 'max_len'],
        [ '4',  4,  4],
        [ '8',  8,  8],
        ['12', 12, 12],
        ['16', 16, 16],
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
    pops, fits = load_runs(**kwargs)
    plot_results(pops, fits, **kwargs)