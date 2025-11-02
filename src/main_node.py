from src.evolve import simulate_tests
from src.genetics import *
from src.utils.plot import plot_results
from src.utils.save import load_fits

kwargs = {
    'name': 'node_demo',
    'seed': None,
    'parallelize': True,
    'saves_path': '../saves/',
    'verbose': 1,  # 0: no updates, 1: generation updates, 2: all updates, 3:
    # Size
    'num_runs': 24,
    'num_gens': 300,
    'pop_size': 100,
    'max_height': 10,  # The maximum height
    # Initialization
    'init_individual_func': random_tree,  # Function used to generate the initial population
    'ops': ['+', '-', '*', '/', '**'],
    'terminals': ['x'],
    'init_max_height': 4,
    'p_branch': 0.5,  # Probability of a node branching
    # Evaluation
    'eval_method': None,
    'target_func': nate,
    'fitness_func': mse,
    'result_fitness_func': mse,  # Fitness to compare results
    'domains': [[-4, 4, 50]],  # The domain of each variable expressed using np.linspace
    # Selection
    'minimize_fitness': True,
    'keep_parents': 2,  # Elitism, must be even
    'k': 2,  # Number of randomly chosen parents for each tournament
    # Repopulation
    'crossover_funcs': [
        [subgraph_crossover, 0.2],
    ],
    'mutate_funcs': [
        [subgraph_mutation, 0.3],
        [pointer_mutation, 0.3],
    ],
    'new_individual_func': random_tree,  # Function used to generate new branches used by mutations
    'subgraph_max_height': 4,
    # Tests
    'test_kwargs': [
        ['Initial Population', 'terminals', ],
        ['Variable Only', ['x'], ],
        # ['With Constants', ['x'] + list(range(-5, 6)), ],
    ],
}

if __name__ == '__main__':
    simulate_tests(**kwargs)
    fits = load_fits(**kwargs)
    plot_results(fits, **kwargs)