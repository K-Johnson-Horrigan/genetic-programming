from numpy import tan

from gp import *

#
# Default kwargs
#

if __name__ == '__main__':

    # kwargs = {
    #     'name': 'sin_to_tan',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 10,
    #     'num_gens': 100,
    #     'pop_size': 1000,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     # 'init_individual_func': random_tree, # Function used to generate the initial population
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x'],
    #     'ops': ['+','-','*','/','**'],
    #     'init_tree_depth': 4,
    #     'target_func': tan,
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'domains': [[-1*np.pi, 1*np.pi, 31]],  # The domain of the problem expressed using np.linspace
    #     'crossover_func': subtree_crossover,
    #     'k': 2, # Number of randomly chosen parents for each tournament
    #     # 'p_c': 0.2, # Probability of crossover
    #     'keep_parents': 2, # Elitism, must be even
    #     # 'mutate_funcs': [
    #     #     [subtree_mutation, 0.3],
    #     #     [pointer_mutation, 0.3],
    #     # ],
    #     'test_kwargs': [
    #         ['Initial Population', 'init_individual_func', 'terminals'       , 'p_c', 'mutate_funcs'                                                      , ],
    #         ['Random Trees'      , random_tree           , ['x','pi','e','i'], 0.6  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #         ['Sin'               , init_sin              , ['x','pi','e','i'], 0.6  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #         ['Limited Sin'       , init_sin_limited      , ['x','pi','e','i'], 0.6  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #     ],
    #     # 'test_kwargs': [
    #     #     ['Initial Population', 'init_individual_func', 'terminals'       , 'p_c', 'mutate_funcs'                                                      , ],
    #     #     ['Random Trees'      , random_tree           , ['x'], 0.6  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #     #     ['Sin'               , init_sin              , ['x'], 0.6  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #     #     ['Limited Sin'       , init_sin_limited      , ['x'], 0.0  , [[point_mutation,0.3],[subtree_mutation,0.3],[pointer_mutation,0.3]], ],
    #     # ],
    # }

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

    #
    # kwargs = {
    #     'name': 'tuning2',
    #     'seed': None,
    #     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 4,
    #     'num_gens': 200,
    #     'pop_size': 100,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'init_individual_func': random_tree, # Function used to generate the initial population
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     'init_tree_depth': 4,
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x', 'i', 'e'],
    #     'ops': ['+', '-', '*', '/', '**'],
    #     'target_func': cos,
    #     'eval_method': None,
    #     'fitness_func': correlation,
    #     'result_fitness_func': mse,  # Fitness to compare results
    #     'domains': [[0, 2 * np.pi, 31]],  # The domain of the problem expressed using np.linspace
    #     'crossover_func': subtree_crossover,
    #     'k': 2,  # Number of randomly chosen parents for each tournament
    #     'keep_parents': 4,  # Elitism, must be even
    #     'test_kwargs': [
    #         ['Probs', 'p_c', 'mutate_funcs'],
    #         *[
    #             [f'Crossover: {p_c}, Subtree: {p_s}, Pointer: {p_p}',
    #                 p_c,
    #                 [
    #                     [subtree_mutation, p_s],
    #                     [pointer_mutation, p_p],
    #                 ]
    #             ]
    #             for p_c in [
    #                 0.125,
    #                 0.75
    #             ]
    #             # for p_m in [0.1, 0.7]
    #             for p_s, p_p in [
    #                 [0.125, 0.125],
    #                 [0.25, 0.25],
    #                 [0.75, 0.125],
    #                 [0.125, 0.75],
    #             ]
    #         ]
    #     ],
    # }


    #
    # kwargs = {
    #     'name': 'noop',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 1,
    #     'num_gens': 100,
    #     'pop_size': 100,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     'init_individual_func': random_noop_tree, # Function used to generate the initial population
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
    #     'p_c': 0.0, # Probability of crossover
    #     'keep_parents': 2, # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.4],
    #         [pointer_mutation, 0.4],
    #     ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.4]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.4],[pointer_mutation, 0.4]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.4],[pointer_mutation, 0.4]]  ],
    #     # ],
    #
    #     'test_kwargs': [
    #         ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #         ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]                          ],
    #         ['DAG'        , random_tree           , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #         ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'num_registers', ],
    #     #     ['1r'         , 1              , ],
    #     #     ['2r'         , 2              , ],
    #     #     ['4r'         , 4              , ],
    #     # ],
    # }





    # kwargs = {
    #     'name': 'sin_to_cos_tuning',
    #     'seed': None,
    #     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 4,
    #     'num_gens': 200,
    #     'pop_size': 100,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'init_individual_func': init_sin_limited,  # Function used to generate the initial population
    #     'new_individual_func': random_tree,  # Function used to generate new branches
    #     'init_tree_depth': 4,
    #     'p_branch': 0.5,  # Probability of a node branching
    #     # 'terminals': ['x', 'i', 'e'],
    #     'terminals': ['x'],
    #     'ops': ['+', '-', '*', '/', '**'],
    #     'target_func': cos,
    #     'eval_method': None,
    #     'fitness_func': correlation,
    #     'result_fitness_func': mse,  # Fitness to compare results
    #     'domains': [[0, 2 * np.pi, 31]],  # The domain of the problem expressed using np.linspace
    #     'crossover_func': subtree_crossover,
    #     'k': 2,  # Number of randomly chosen parents for each tournament
    #     'keep_parents': 4,  # Elitism, must be even
    #     'test_kwargs': [
    #         ['Probabilities', 'p_c', 'mutate_funcs'],
    #         *[
    #             [f'Crossover: {p_c}, Subtree: {p_sub}, Point: {p_point}, Pointer: {p_pointer}',
    #                 p_c,
    #                 [
    #                     [subtree_mutation, p_sub],
    #                     [point_mutation,   p_point],
    #                     [pointer_mutation, p_pointer],
    #                 ]
    #             ]
    #             for p_c in [
    #                 2/8,
    #                 6/8,
    #             ]
    #             for p_sub, p_point, p_pointer in [
    #                 [2/8, 2/8, 2/8],
    #                 [0/8, 3/8, 3/8],
    #                 [3/8, 0/8, 3/8],
    #                 [3/8, 3/8, 0/8],
    #             ]
    #         ]
    #     ],
    # }

    # kwargs = {
    #     'name': 'sin_to_tan_test',
    #     'seed': None,
    #     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 1,
    #     'num_gens': 100,
    #     'pop_size': 1000,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree,  # Function used to generate new branches
    #     'init_individual_func': random_tree,  # Function used to generate the initial population
    #     'num_registers': 2,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5,  # Probability of a node branching
    #     # 'terminals': ['x','i','e','pi'],
    #     'terminals': ['x_0', 'x_1', 'x_2', 'x_3'],
    #     'ops': ['+', '-', '*', '/', '**'],
    #     'target_func': xor_and_xor,
    #     # 'domains': [[-np.pi, np.pi, 31]],  # The domain of the problem expressed using np.linspace
    #     'domains': [(0, 1, 2)]*4,  # The domain of the problem expressed using np.linspace
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse,  # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2,  # Number of randomly chosen parents for each tournament
    #     'p_c': 0.1,  # Probability of crossover
    #     'keep_parents': 2,  # Elitism, must be even
    #     # 'mutate_funcs': [
    #     #     [subtree_mutation, 0.75],
    #     #     [point_mutation, 0.25],
    #     # ],
    #
    #     'test_kwargs': [
    #         ['Mutation', 'p_c', 'mutate_funcs'],
    #         *[
    #             [f'{pc} {st} {pt} {pr} {ss} {ds}', pc, [[subtree_mutation, st], [point_mutation, pt], [pointer_mutation, pr], [split_mutation, ss], [deep_split_mutation, ds]]]
    #             for pc in [0.6]
    #             for st in [.3]
    #             for pt in np.array([.3]) * [1]
    #             for pr in np.array([.3]) * [1]
    #             for ss in [0]
    #             for ds in [0]
    #             if st + pt + pr + ss + ds <= 1
    #         ]
    #     ],
    # }



    kwargs = {
        'name': 'bit_sum',
        'seed': None,
        'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
        'num_runs': 10,
        'num_gens': 50,
        'pop_size': 500,
        'max_tree_depth': 23,
        'max_subtree_depth': 4,
        'eval_method': None,
        'new_individual_func': random_tree,  # Function used to generate new branches
        'init_individual_func': init_get_bit,  # Function used to generate the initial population
        'num_registers': 2,
        'init_tree_depth': 2,
        'p_branch': 0.5,  # Probability of a node branching
        'terminals': ['x'],
        'ops': ['+', '-', '*', '/', '**'],
        'target_func': bit_sum,
        'domains': [(0, 15, 16)],  # The domain of the problem expressed using np.linspace
        # 'fitness_func': correlation,
        'fitness_func': mse,
        'result_fitness_func': mse,  # Fitness to compare results
        'crossover_func': subtree_crossover,
        'k': 2,  # Number of randomly chosen parents for each tournament
        'p_c': 0.1,  # Probability of crossover
        'keep_parents': 2,  # Elitism, must be even
        'mutate_funcs': [
            [subtree_mutation, 0.1],
            [point_mutation,   0.4],
            [pointer_mutation, 0.4],
        ],

        'test_kwargs': [
            ['Initial Population', 'init_individual_func', ],
            ['Random Trees'      , random_tree           , ],
            ['Get Bits'          , init_get_bit          , ],
            ['Limited Get Bits'  , init_get_bit_limited  , ],
        ],
    }


    # Run simulation, save, then plot
    all_pops, all_fits = simulate_tests(**kwargs)
    save_all(all_pops, all_fits, kwargs)
    plot_results(all_pops, all_fits, **kwargs)