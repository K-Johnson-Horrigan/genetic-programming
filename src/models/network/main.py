from src.evolve import simulate_tests
from src.utils.plot import plot_results
from src.utils.save import load_runs, load_fits


kwargs = {
    'name': 'test_0',  # Name of folder to contain all results
    'seed': None,
    'verbose': True,
    'parallelize': True,
    'saves_path': '../../../saves/network/',  # Save path relative to this file
    ## Size ##
    'num_runs': 12,
    'num_gens': 100,
    'pop_size': 100,
    ## Initialization ##
    'init_individual_func': random_code,  # Function used to generate a new organism
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
    'test_kwargs': [
        ['Ops', 'ops'],
        ['Normal', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ',)],
        ['DEL', ('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ', 'DEL',)],
    ],
}

if __name__ == '__main__':
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)