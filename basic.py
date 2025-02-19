import random
import string
from evolve import *
from utils import *

# All functions relevant to non-genetic programming.

#
# Initialization
#

def gen_individual(target, **kwargs):
    return ''.join(random.choices(string.ascii_uppercase, k=len(target)))

#
# Evaluation
#

def fitness_func(pop, target, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    return [sum([target[i] != letter for i, letter in enumerate(individual)]) for individual in pop]

#
# Mutation
#

def mutation_func(individual, p_m, verbose=0, **kwargs):
    """Perform a mutation with a probability of p_m"""
    return ''.join([random.choice(string.ascii_uppercase) if random.random() < p_m else l for l in individual])

#
# Default kwargs
#

kwargs = {
    'seed': None,
    'num_runs': 10,
    'num_gens': 200,
    'pop_size': 1,
    'verbose': 2,
    'lambda': 20,
    'keep_parents': True,
    'gen_individual': gen_individual,
    'fitness_func': fitness_func,
    'target': 'PROGRAMMING',
    'mutate_func': mutation_func,
    'p_m': 0.4, # Probability of a bit mutating
}

if __name__ == '__main__':

    # kwargs['label_title'] = 'Probability of Mutation'
    # kwargs['labels'] = kwargs['values'] = [0.3, 0.4, 0.5, 0.7]
    # kwargs['key'] = 'p_m'

    kwargs['label_title'] = kwargs['key'] = 'lambda'
    kwargs['labels'] = kwargs['values'] = [10, 20, 30]
    all_pops, all_fits = run_sims(**kwargs)
    plot_sims(all_pops, all_fits, **kwargs)

    kwargs['label_title'] = kwargs['key'] = 'keep_parents'
    kwargs['labels'] = kwargs['values'] = [True, False]
    all_pops, all_fits = run_sims(**kwargs)
    plot_sims(all_pops, all_fits, **kwargs)
