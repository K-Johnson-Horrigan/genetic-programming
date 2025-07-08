from evolve import simulate_tests
from src.utils.plot import plot_results
from src.utils.save import load_runs, save_kwargs, load_kwargs
from genetics import *

# kwargs = {
#     'name': 'debug',
#     'seed': None,
#     'verbose': 1, # 0: no updates, 1: generation updates, 2: all updates, 3:
#     'parallelize': True,
#     # Size
#     'num_runs': 2,
#     'num_gens': 100,
#     'pop_size': 400,
#     # 'max_height': 10, # The maximum height
#     'tape_dim': 2, # Dimensionality of the Turing tape
#     'tm_timeout': 100, # Number of TM iterations before forcing a halting state
#     # Initialization
#     'init_individual_func': random_trans, # Function used to generate the initial population
#     'init_min_len': 3,
#     'init_max_len': 5,
#     'states': ['start', 'halt'] + [str(i) for i in range(5)],
#     'symbols': list(range(5)),
#     'moves': [-1,0,1],
#     # Evaluation
#     'fitness_func': diff_zeros,
#     # 'target': [
#     #     [1,0,0],
#     #     [2,0,0],
#     #     [3,2,1],
#     # ],
#     'target': [
#         [1, 0, 0],
#         [2, 0, 0],
#         [3, 2, 1],
#     ],
#     'minimize_fitness': True,
#     # Selection
#     'keep_parents': 2, # Elitism, must be even
#     'k': 2, # Number of randomly chosen parents for each tournament
#     # Repopulation
#     'p_c': 0.2, # Probability of crossover
#     'crossover_func': two_point_crossover,
#     # 'subgraph_max_height': 4,
#     'mutate_funcs': [
#         [point_mutation, 0.9],
#     ],
#     # Tests
#     'test_kwargs': [
#         ['Initial Population', 'p_c',],
#         ['Low', 0.7,],
#         ['High', 0.9,],
#         # ['With Constants', ['x']+list(range(-5,6)),],
#     ],
# }

# kwargs = {
#     'name': 'tuning_1',
#     'seed': None,
#     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
#     'parallelize': True,
#     # Size
#     'num_runs': 1,
#     'num_gens': 100,
#     'pop_size': 100,
#     'tape_dim': 2,  # Dimensionality of the Turing tape
#     'tm_timeout': 100,  # Number of TM iterations before forcing a halting state
#     'head_shape': (2, 2),
#     # Initialization
#     'init_individual_func': random_trans,  # Function used to generate the initial population
#     'init_min_len': 1,
#     'init_max_len': 4,
#     'states': ['start', 'halt'] + [str(i) for i in range(5)],
#     'symbols': list(range(5)),
#     # 'moves': [-2, -1, 0, 1, 2],
#     'moves': [-1, 0, 1],
#     # Evaluation
#     'fitness_func': diff_zeros,
#     # 'target': [
#     #     [1,0,0],
#     #     [2,0,0],
#     #     [3,2,1],
#     # ],
#     'target': [
#         [1, 1, 0, 0, 1, 1],
#         [1, 1, 0, 0, 1, 1],
#         [0, 0, 1, 1, 0, 0],
#         [0, 1, 1, 1, 1, 0],
#         [0, 1, 1, 1, 1, 0],
#         [0, 1, 0, 0, 1, 0],
#     ],
#     'minimize_fitness': True,
#     # Selection
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     # Repopulation
#     'p_c': 0.7,  # Probability of crossover
#     'crossover_func': one_point_crossover,
#     'mutate_funcs': [
#         [point_mutation, 0.7],
#     ],
#     # Tests
#     # 'test_kwargs': [
#     #     ['Crossover, Mutation', 'p_c', 'mutate_funcs'],
#     #     *[
#     #         [f'{pc} {pt}', pc, [[point_mutation, pt]]]
#     #         for pc in [.3,.5,.7,.9]
#     #         for pt in [.3,.5,.7,.9]
#     #     ]
#     # ],
#     'test_kwargs': [
#         ['States, Symbols', 'states', 'symbols'],
#         *[
#             [f'{states} {symbols}', ['start', 'halt'] + [str(i) for i in range(states)], [-1] + list(range(symbols))]
#             for states in range(0, 4, 1)
#             for symbols in range(2, 6, 1)
#         ]
#     ],
# }

# spiral = np.array([
#     [1,1,1,1,1,1,1,1,1,1,1],
#     [1,0,0,0,0,0,0,0,0,0,1],
#     [1,1,1,1,1,1,1,1,1,0,1],
#     [1,0,1,0,0,0,0,0,0,0,1],
#     [1,0,1,0,1,1,1,1,1,0,1],
#     [1,0,1,0,1,0,0,0,1,0,1],
#     [1,0,1,0,1,1,1,0,1,0,1],
#     [1,0,1,0,0,0,0,0,1,0,1],
#     [1,0,1,1,1,1,1,1,1,0,1],
#     [1,0,0,0,0,0,0,0,0,0,1],
#     [1,1,1,1,1,1,1,1,1,1,1],
# ]).T

# kwargs = {
#     'name': 'array_maze_spiral_0',  # Folder to contain all results
#     'seed': None,
#     'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
#     'parallelize': True,
#     # Size
#     'num_runs': 16,
#     'num_gens': 200,
#     'pop_size': 10, #1000
#     # Turing Machine Specifications
#     'tape_dim': 2,  # Dimensionality of the Turing tape
#     'tm_timeout': 200,  # Number of TM iterations before forcing a halting state
#     'head_shape': (1, 1),
#     'states': ['start'] + [str(i) for i in range(1)],
#     'symbols': list(range(5)),
#     'moves': [-1, 0, 1],
#     # Initialization
#     'init_individual_func': random_trans,  # Function used to generate the initial population
#     'init_min_len': 5,
#     'init_max_len': 10,
#     # Evaluation
#     'fitness_func': maze_fitness,
#     'target': format_maze(gen_spiral_maze((15, 15))),
#     # 'target': _format_maze(maze),
#     'minimize_fitness': False,
#     # Selection
#     'keep_parents': 2,  # Elitism, must be even
#     'k': 2,  # Number of randomly chosen parents for each tournament
#     # Repopulation
#     'p_c': 0.7,  # Probability of crossover
#     'crossover_func': one_point_crossover,
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
#             for pc in [.7,]
#             for pt in [.5,]
#         ]
#     ],
#     # 'test_kwargs': [
#     #     ['States, Symbols', 'states', 'symbols'],
#     #     *[
#     #         [f'{states} {symbols}', ['start'] + [str(i) for i in range(states)],
#     #          [TM.ANY] + list(range(symbols))]
#     #         # for states in range(0, 4, 1)
#     #         # for symbols in range(2, 6, 1)
#     #         for states in [1]
#     #         for symbols in [2]
#     #     ]
#     # ],
# }

maze = format_maze(gen_maze((15,15)))
# maze = format_maze(gen_spiral_maze((11,11)))
maze[maze==5] = -1
maze[maze>0] = 1
maze[maze==-1] = 5

kwargs = {
    'name': 'array_maze_mono_massive',  # Folder to contain all results
    'seed': None,
    'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
    'parallelize': True,
    'saves_path': '../saves/',
    # Size
    'num_runs': 20,
    'num_gens': 200,
    'pop_size': 400,
    # 'num_gens': 20,
    # 'pop_size': 40,
    # Turing Machine Specifications
    'tape_dim': 2,  # Dimensionality of the Turing tape
    'tm_timeout': 400,  # Number of TM iterations before forcing a halting state
    'states': list(range(2)),
    'symbols': list(range(5)),
    'moves': [-1, 0, 1],
    # Initialization
    'init_individual_func': random_trans_array,  # Function used to generate the initial population
    # Evaluation
    'fitness_func': maze_fitness,
    # 'target': format_maze(gen_maze((11,11))),
    'target': maze,
    # Selection
    'minimize_fitness': False,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    # Repopulation
    'p_c': 0.0,  # Probability of crossover
    'crossover_func': 0,
    'mutate_funcs': [
        [macro_mutation, 0.7],
    ],
    # Tests
    # 'test_kwargs': [['Examples'],['Example 0']],
    # 'test_kwargs': [
    #     ['Crossover, Mutation', 'p_c', 'mutate_funcs'],
    #     *[
    #         [f'{pc} {pt}', pc, [[macro_mutation, pt]]]
    #         # for pc in [.3,.5,.7,.9]
    #         # for pt in [.3,.5,.7,.9]
    #         for pc in [0,]
    #         for pt in [.5,]
    #     ]
    # ],
    'test_kwargs': [
        ['States, Symbols', 'states', 'symbols'],
        *[
            [f'{states} {symbols}', list(range(states)), list(range(symbols))]
            # for states in [2]
            # for symbols in [2]
            for states in [2,3,4,5]
            for symbols in [2,3,4,5]
        ]
    ],
}


if __name__ == '__main__':
    # This replaces extra values in the formated maze
    kwargs['maze_sol'] = solve_maze((kwargs['target'] != 0) * 1)
    simulate_tests(**kwargs)
    # kwargs['saves_path'] = '../../saves/'
    # kwargs = load_kwargs(kwargs['name'], kwargs['saves_path'])
    pops, fits = load_runs(**kwargs)
    plot_results(pops, fits, **kwargs)