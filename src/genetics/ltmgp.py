"""
Genetic programming functions specifically for the evolution of linear Turing Machines.
Turing Machines are represented as 2D arrays and are converted to TM objects when evaluating.
"""

from src.utils.utils import choice

#
# Initialization Functions
#

def _random_transition(**kwargs):
    """Helper function for generating only a single transition"""
    return [
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        choice(kwargs['states'], kwargs['rng']),
        choice(kwargs['symbols'], kwargs['rng']),
        [choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
    ]

# def _random_transition(**kwargs):
#     """Helper function for generating only a single transition"""
#     return [
#         choice(kwargs['states'], kwargs['rng']),
#         to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
#         choice(kwargs['states'], kwargs['rng']),
#         to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape'])),
#         *[choice(kwargs['moves'], kwargs['rng']) for _ in range(kwargs['tape_dim'])]
#     ]


def random_trans(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    trans = [_random_transition(**kwargs) for _ in range(init_len)]
    return trans


#
# Fitness Functions
#


#
# Mutation Functions
#

# def point_mutation(trans, **kwargs):
#     """Randomly change a value in a random transition"""
#     # Duplicate the original transitions
#     trans = [t.copy() for t in trans]
#     # Select a random transition
#     index = kwargs['rng'].integers(len(trans))
#     t = trans[index]
#     # Select a random argument within the transition
#     sub_index = kwargs['rng'].integers(len(t))
#     # Change the sub parameter
#     if sub_index == 0 or sub_index == 2:
#         trans[index][sub_index] = choice(kwargs['states'], kwargs['rng'])
#     elif sub_index == 1 or sub_index == 3:
#         trans[index][sub_index] = choice(kwargs['symbols'], kwargs['rng'])
#         # trans[index][sub_index] = to_tuple(kwargs['rng'].choice(kwargs['symbols'], kwargs['head_shape']))
#     else:
#         trans[index][sub_index] = [choice(kwargs['moves'], kwargs['rng']) for i in range(len(t[4]))]
#     return trans


#
# Crossover Functions
#


#
# Target Functions
#


#
# Initial pops
#


#
# Debug
#

if __name__ == '__main__':
    pass