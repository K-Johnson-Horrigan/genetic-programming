from gp import *

#
# Default kwargs
#

if __name__ == '__main__':

    # kwargs = {
    #     'name': 'cos',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_reps': 1,
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


    kwargs = {
        'name': 'tuning',
        'seed': None,
        'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
        'num_reps': 1,
        'num_gens': 200,
        'pop_size': 100,
        'max_tree_depth': 100,
        'max_subtree_depth': 4,
        'init_individual_func': random_tree, # Function used to generate the initial population
        'new_individual_func': random_tree, # Function used to generate new branches
        'init_tree_depth': 4,
        'p_branch': 0.5, # Probability of a node branching
        'terminals': ['x', 'i', 'e'],
        'ops': ['+', '-', '*', '/', '**'],
        'target_func': cos,
        'eval_method': None,
        'fitness_func': correlation,
        'result_fitness_func': mse,  # Fitness to compare results
        'domains': [[0, 2 * np.pi, 63]],  # The domain of the problem expressed using np.linspace
        'crossover_func': subtree_crossover,
        'k': 4,  # Number of randomly chosen parents for each tournament
        'keep_parents': 4,  # Elitism, must be even
        'test_kwargs': [
            ['Probs', 'p_c', 'mutate_funcs'],
            *[
                [f'{p_c}c {p_s}s {p_p}p',
                    p_c,
                    [
                        [subtree_mutation, p_s],
                        [pointer_mutation, p_p],
                    ]
                ]
                for p_c in np.arange(5, 11, 2) / 10
                for p_s in np.arange(5, 11, 2) / 10
                for p_p in (1 - p_s) * (np.arange(0, 2, 1))
            ]
        ],
    }



    # kwargs = {
    #     'name': 'noop',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_reps': 1,
    #     'num_gens': 100,
    #     'pop_size': 100,
    #     'max_tree_depth': 16,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     # 'init_individual_func': random_tree, # Function used to generate the initial population
    #     'num_registers': 4,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x', 'e', 'i'],
    #     'ops': ['+','-','*','/','**'],
    #     'target_func': cos,
    #     'domains': [[0, 2*np.pi, 63]],  # The domain of the problem expressed using np.linspace
    #     'fitness_func': correlation,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2, # Number of randomly chosen parents for each tournament
    #     'p_c': 0.9, # Probability of crossover
    #     'keep_parents': 4, # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.5],
    #         [pointer_mutation, 0.5],
    #     ],
    #     'test_kwargs': [
    #         ['Initial Pop', 'init_individual_func', 'mutate_funcs'],
    #         ['Tree', random_tree,      [[subtree_mutation, 0.5]]],
    #         ['DAG',  random_tree,      [[subtree_mutation, 0.5],[pointer_mutation, 0.5]]  ],
    #         ['LGP',  random_noop_tree, [[subtree_mutation, 0.5],[pointer_mutation, 0.5]]  ],
    #     ],
    # }






    # Run simulation, save, then plot
    all_pops, all_fits = run_sims(**kwargs)
    save_all(all_pops, all_fits, kwargs)
    plot_results(all_pops, all_fits, **kwargs)