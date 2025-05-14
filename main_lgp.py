from gp import *
from gp import correlation

#
# Default kwargs
#

if __name__ == '__main__':

    # kwargs = {
    #     'name': 'noop',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 1,
    #     'num_gens': 300,
    #     'pop_size': 100,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     'init_individual_func': random_noop_tree, # Function used to generate the initial population
    #     'num_registers': 2,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x', 'e', 'i'],
    #     'ops': ['+','-','*','/','**'],
    #     'target_func': cos,
    #     'domains': [[0, 2*np.pi, 31]],  # The domain of the problem expressed using np.linspace
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2, # Number of randomly chosen parents for each tournament
    #     'p_c': 0.7, # Probability of crossover
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
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     # ],
    #
    #     'test_kwargs': [
    #         ['Initial Pop', 'num_registers', ],
    #         ['1r'         , 1              , ],
    #         ['2r'         , 2              , ],
    #         ['4r'         , 4              , ],
    #         ['8r'         , 8              , ],
    #     ],
    # }


    # kwargs = {
    #     'name': 'registers_redo',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 1,
    #     'num_gens': 300,
    #     'pop_size': 300,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     'init_individual_func': random_noop_tree, # Function used to generate the initial population
    #     'num_registers': 2,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x'],
    #     'ops': ['+','-','*','/','**'],
    #     'target_func': k3,
    #     'domains': [[-1, 1, 50]],  # The domain of the problem expressed using np.linspace
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2, # Number of randomly chosen parents for each tournament
    #     'p_c': 0.5, # Probability of crossover
    #     'keep_parents': 2, # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.4],
    #         # [pointer_mutation, 0.4],
    #     ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.4]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.4],[pointer_mutation, 0.4]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.4],[pointer_mutation, 0.4]]  ],
    #     # ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     # ],
    #
    #     'test_kwargs': [
    #         ['Initial Pop', 'num_registers', ],
    #         ['1r'         , 1              , ],
    #         ['2r'         , 2              , ],
    #         ['4r'         , 4              , ],
    #         ['8r'         , 8              , ],
    #     ],
    # }




    # kwargs = {
    #     'name': 'noop4',
    #     'seed': None,
    #     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 5,
    #     'num_gens': 300,
    #     'pop_size': 200,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree, # Function used to generate new branches
    #     'init_individual_func': random_noop_tree, # Function used to generate the initial population
    #     'num_registers': 4,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5, # Probability of a node branching
    #     'terminals': ['x'],
    #     'ops': ['+','-','*','/','**'],
    #     'target_func': k3,
    #     'domains': [[-1, 1, 50]],  # The domain of the problem expressed using np.linspace
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse, # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2, # Number of randomly chosen parents for each tournament
    #     'p_c': 0.7, # Probability of crossover
    #     'keep_parents': 2, # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.7],
    #         # [pointer_mutation, 0.4],
    #     ],
    #
    #     'test_kwargs': [
    #         ['Mutation', 'init_individual_func', 'mutate_funcs'                                     ],
    #         ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]],
    #         ['No Split'        , random_tree           , [[subtree_mutation, 0.4],[pointer_mutation, 0.4]]],
    #         ['Shallow Split'        , random_tree           , [[subtree_mutation, 0.2],[pointer_mutation, 0.3],[split_mutation, 0.3]]],
    #         ['Deep Split'        , random_tree      , [[subtree_mutation, 0.2],[pointer_mutation, 0.3],[deep_split_mutation, 0.3]]],
    #     ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     # ],
    # }



    # kwargs = {
    #     'name': 'noop5',
    #     'seed': None,
    #     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates
    #     'num_runs': 5,
    #     'num_gens': 300,
    #     'pop_size': 200,
    #     'max_tree_depth': 10,
    #     'max_subtree_depth': 4,
    #     'eval_method': None,
    #     'new_individual_func': random_tree,  # Function used to generate new branches
    #     'init_individual_func': random_noop_tree,  # Function used to generate the initial population
    #     'num_registers': 4,
    #     'init_tree_depth': 2,
    #     'p_branch': 0.5,  # Probability of a node branching
    #     'terminals': ['x'],
    #     'ops': ['+', '-', '*', '/', '**'],
    #     'target_func': k3,
    #     'domains': [[-1, 1, 50]],  # The domain of the problem expressed using np.linspace
    #     # 'fitness_func': correlation,
    #     'fitness_func': mse,
    #     'result_fitness_func': mse,  # Fitness to compare results
    #     'crossover_func': subtree_crossover,
    #     'k': 2,  # Number of randomly chosen parents for each tournament
    #     'p_c': 0.7,  # Probability of crossover
    #     'keep_parents': 2,  # Elitism, must be even
    #     'mutate_funcs': [
    #         [subtree_mutation, 0.7],
    #         # [pointer_mutation, 0.4],
    #     ],
    #
    #     'test_kwargs': [
    #         ['Mutation', 'init_individual_func', 'mutate_funcs'],
    #         ['Tree', random_tree, [[subtree_mutation, 0.8]]],
    #         ['No Split', random_tree, [[subtree_mutation, 0.0], [pointer_mutation, 0.8]]],
    #         ['Shallow Split', random_tree, [[subtree_mutation, 0.0], [pointer_mutation, 0.6], [split_mutation, 0.2]]],
    #         ['Deep Split', random_tree, [[subtree_mutation, 0.0], [pointer_mutation, 0.6], [deep_split_mutation, 0.2]]],
    #     ],
    #
    #     # 'test_kwargs': [
    #     #     ['Initial Pop', 'init_individual_func', 'mutate_funcs'                                     ],
    #     #     ['Tree'       , random_tree           , [[subtree_mutation, 0.8]]                          ],
    #     #     ['DAG'        , random_tree           , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     #     ['LGP'        , random_noop_tree      , [[subtree_mutation, 0.1],[pointer_mutation, 0.7]]  ],
    #     # ],
    # }




    kwargs = {
        'name': 'registers1',
        'seed': None,
        'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates
        'num_runs': 10,
        'num_gens': 200,
        'pop_size': 100,
        'max_tree_depth': 10,
        'max_subtree_depth': 4,
        'eval_method': None,
        'new_individual_func': random_tree, # Function used to generate new branches
        'init_individual_func': random_noop_tree, # Function used to generate the initial population
        'num_registers': 2,
        'init_tree_depth': 2,
        'p_branch': 0.5, # Probability of a node branching
        'terminals': ['x'],
        'ops': ['+','-','*','/','**'],
        'target_func': k3,
        'domains': [[-1, 1, 50]],  # The domain of the problem expressed using np.linspace
        # 'fitness_func': correlation,
        'fitness_func': mse,
        'result_fitness_func': mse, # Fitness to compare results
        'crossover_func': subtree_crossover,
        'k': 2, # Number of randomly chosen parents for each tournament
        'p_c': 0.1, # Probability of crossover
        'keep_parents': 2, # Elitism, must be even
        # 'mutate_funcs': [
        #     [subtree_mutation, 0.75],
        #     [point_mutation, 0.25],
        # ],

        # 'test_kwargs': [
        #     ['Mutation', 'init_individual_func', 'mutate_funcs'],
        #     ['Tree'          , random_noop_tree, [[subtree_mutation, 2/8]]],
        #     ['No Split'      , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3]]],
        #     ['Shallow Split' , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3],[split_mutation,      1/8]]],
        #     ['Deep Split'    , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3],[deep_split_mutation, 1/8]]],
        # ],

        # 'test_kwargs': [
        #     ['Mutation', 'init_individual_func', 'mutate_funcs'],
        #     ['Tree'           , random_noop_tree, [[subtree_mutation, 7/8]]],
        #     ['No Split Macro' , random_noop_tree, [[subtree_mutation, 3/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3]]],
        #     ['No Split Micro' , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 5/8*1/3],[pointer_mutation, 5/8*2/3]]],
        #     ['Shallow Split'  , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3],[split_mutation,      1/8]]],
        #     ['Deep Split'     , random_noop_tree, [[subtree_mutation, 2/8],[point_mutation, 4/8*1/3],[pointer_mutation, 4/8*2/3],[deep_split_mutation, 1/8]]],
        # ],

        'test_kwargs': [
            ['Mutation', 'p_c', 'num_registers', 'mutate_funcs'],
            *[
                [f'{pc} {nr} {st} {pt} {pr} {ss} {ds}', pc, nr, [[subtree_mutation, st], [point_mutation, pt], [pointer_mutation, pr], [split_mutation, ss], [deep_split_mutation, ds]]]
                for pc in [0.1]
                for nr in [1,2,4,8]
                for st in [2/8]
                for pt in [4/8*2/3]
                for pr in [4/8*1/3]
                for ss in [0]
                for ds in [0]
                if st + pt + pr + ss + ds <= 1
            ]
        ],
    }



    # Run simulation, save, then plot
    all_pops, all_fits = simulate_tests(**kwargs)
    save_all(all_pops, all_fits, kwargs)
    plot_results(all_pops, all_fits, **kwargs)