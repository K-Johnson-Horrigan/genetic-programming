from gp import *

if __name__ == '__main__':

    kwargs = {
        'name': 'debug',
        'seed': None,
        'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
        'num_runs': 2,
        'num_gens': 100,
        'pop_size': 100,
        'max_tree_depth': 10,
        'max_subtree_depth': 4,
        'eval_method': None,
        'new_individual_func': random_tree, # Function used to generate new branches
        'init_individual_func': random_tree, # Function used to generate the initial population
        'p_branch': 0.5, # Probability of a node branching
        'terminals': ['x'],
        'ops': ['+','-','*','/','**'],
        'init_tree_depth': 4,
        'target_func': k3,
        # 'fitness_func': correlation,
        'fitness_func': mse,
        'result_fitness_func': mse, # Fitness to compare results
        'domains': [[-1, 1, 50]],  # The domain of the problem expressed using np.linspace
        'crossover_func': subtree_crossover,
        'k': 2, # Number of randomly chosen parents for each tournament
        'p_c': 0.2, # Probability of crossover
        'keep_parents': 2, # Elitism, must be even
        'mutate_funcs': [
            [subtree_mutation, 0.3],
            [pointer_mutation, 0.3],
        ],
        'test_kwargs': [
            ['Initial Population', 'terminals',],
            ['Just x', ['x'],],
            ['Numbers', ['x',1,2,3],],
        ],
    }

    # kwargs = {
    #     'name': 'cos',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 1,
    #     'num_gens': 10,
    #     'pop_size': 60,
    #     'max_tree_depth': 200,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     # 'init_individual_func': random_tree, # Function used to generate the initial population
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x'],
    #     'ops': ['+','-','*','/','**'],
    #     'init_tree_depth': 4,
    #     'target_func': cos,
    #     'fitness_func': correlation,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'domains': [[0, 2*np.pi, 63]],  # The domain of the problem expressed using np.linspace
    #     'crossover_func': subtree_crossover,
    #     'k': 4, # Number of randomly chosen parents for each tournament
    #     'p_c': 0.9, # Probability of crossover
    #     'keep_parents': 4, # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.3],
    #         [pointer_mutation, 0.3],
    #     ],
    #     'test_kwargs': [
    #         ['Initial Pop', 'init_individual_func',],
    #         ['random', random_tree],
    #         ['init sin', init_sin],
    #         ['init sin limited', init_sin_limited],
    #     ],
    # }

    # Run simulation, save, then plot
    simulate_tests(**kwargs)
    # save_all(all_pops, all_fits, kwargs)
    # plot_results(all_pops, all_fits, **kwargs)