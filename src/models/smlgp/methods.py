"""
Genetic programming functions specifically for the evolution of linear models.
Linear code is represented as a 2D arrays and converted to a Linear objects when evaluating.
"""
import numpy as np

from src.models.smlgp.model import Linear
from src.utils.utils import cartesian_prod


#
# Initialization Functions
#

def _random_line(**kwargs):
    """Helper function to generating a single line of code"""
    return [
        # kwargs['rng'].choice(kwargs['ops']),
        kwargs['rng'].integers(kwargs['max_value']),
        kwargs['rng'].integers(kwargs['max_len']),
        kwargs['rng'].integers(kwargs['max_value']),
        # kwargs['rng'].integers(1+2*len(kwargs['mem_lens'])),
        kwargs['rng'].integers(3),
    ]


def random_mem(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len']//Linear.LINE_LENGTH, kwargs['init_max_len']//Linear.LINE_LENGTH+1)
    code = [_random_line(**kwargs) for _ in range(init_len)]
    code = sum(code, [])  # Flatten list of lists
    return code


def random_mems(**kwargs):
    """Generate a random list of mems"""
    code = []
    for i in range(len(kwargs['init_max_lens'])):
        init_min_len = kwargs['init_min_lens'][i]
        init_max_len = kwargs['init_min_lens'][i]
        code.append(random_mem(init_min_len=init_min_len, init_max_len=init_max_len, **kwargs))
    return code

# def random_code(**kwargs):
#     self_rep = [_random_line(**kwargs) for _ in range(6)]
#     self_rep = sum(self_rep, [])
#     init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
#     code = [_random_line(**kwargs) for _ in range(init_len)]
#     code = sum(code, [])
#     code = [code, self_rep]
#     return code

def random_code_uniform(**kwargs):
    self_rep = [
        Linear.RAND,  2,  1, Linear.VARS_DIRECT,   # Generate random value
        Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
        Linear.LOAD,  3,  2, Linear.MEM2_INDIRECT, # Load temp value from MEM2
        Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
        Linear.STORE, 3,  2, Linear.MEM3_INDIRECT, # Store temp value into MEM2
        Linear.ADD,   2,  1, Linear.IMMEDIATE,     # Increment copy pointer
    ]
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    code = [_random_line(**kwargs) for _ in range(init_len)]
    code = sum(code, [])
    code = [code, self_rep]
    return code

def random_code_one_point(**kwargs):
    self_rep = [
        Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Check if copy pointer is 0
        Linear.RAND,4*7,  1, Linear.VARS_DIRECT,   # Randomly move the copy pointer
        Linear.LOAD,  2,  1, Linear.MEM2_INDIRECT, # Load temp value from MEM2
        Linear.STORE, 2,  1, Linear.MEM3_INDIRECT, # Store temp value into MEM3
        Linear.ADD,   1,  1, Linear.IMMEDIATE,     # Increment copy pointer
        Linear.IFEQ,  1,4*7, Linear.IMMEDIATE,     # Check if copy pointer is at last position
        Linear.STOP,  0,  0, Linear.IMMEDIATE,     # End execution
    ]
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    code = [_random_line(**kwargs) for _ in range(init_len)]
    code = sum(code, [])
    code = [code, self_rep]
    return code

# def random_random_code(**kwargs):
#     func_index = kwargs['rng'].integers(0, 3)
#     funcs = [random_code_uniform, random_code_one_point, random_non_self_rep_code]
#     return funcs[func_index](**kwargs)

#
# Fitness Functions
#

def run_self_rep(code, **kwargs):
    """Run the given code setup to self replicate"""
    l = Linear([[0]*4, code, [-1]*len(code)])
    l.run(kwargs['timeout'])
    return l


def self_rep(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population"""
    fits = np.empty(len(pop))
    for i, code in enumerate(pop):
        code_1d = np.ravel(code)
        l = run_self_rep(code_1d, **kwargs)
        fit = sum(code_1d != l.mem[2])
        fits[i] = fit
    return fits


# def self_mutate(pop, **kwargs):
#     """Calculate the fitness value of all individuals in a population"""
#     fits = np.empty(len(pop))
#     for i,code in enumerate(pop):
#
#         input_0 = [0] * 4
#         input_1 = [0] * 4
#         l_0 = Linear(code, [0], 4)
#         l_1 = Linear(code, [0], 4)
#         l_0.run(kwargs['timeout'])
#         l_1.run(kwargs['timeout'])
#         output_0 = l_0.out
#         output_1 = l_1.out
#         output_0 = np.array(output_0)
#         output_1 = np.array(output_1)
#
#         diff_0 = input_0 != output_0
#         diff_1 = input_1 != output_1
#
#         fit = 0
#         fit += sum(diff_0)
#         fit += sum(diff_1)
#
#         if sum(diff_0) + sum(diff_1) == 0:
#             fit = 1000
#
#         if (output_0 == output_1).all():
#             fit = 1000
#
#         fits[i] = fit
#     return fits


# def self_crossover(pop, **kwargs):
#     """Calculate the fitness value of all individuals in a population"""
#     fits = np.empty(len(pop))
#     for i, code in enumerate(pop):
#
#         input_0 = [0] * 4
#         input_1 = [0] * 4
#         l_0 = Linear(code, [0], 4)
#         l_1 = Linear(code, [0], 4)
#         l_0.run(kwargs['timeout'])
#         l_1.run(kwargs['timeout'])
#         output_0 = l_0.out
#         output_1 = l_1.out
#         output_0 = np.array(output_0)
#         output_1 = np.array(output_1)
#
#         diff_0 = input_0 != output_0
#         diff_1 = input_1 != output_1
#
#         fit = 0
#         fit += sum(diff_0)
#         fit += sum(diff_1)
#
#         if sum(diff_0) + sum(diff_1) == 0:
#             fit = 1000
#
#         if (output_0 == output_1).all():
#             fit = 1000
#
#         fits[i] = fit
#     return fits

def lgp_rmse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    return lgp_mse(pop, target_func, domains, rmse=True, **kwargs)


def lgp_mse(pop, target_func, domains, rmse=False, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    # 2D array of input variables for each test case
    cases = cartesian_prod(*domains)
    y_target = np.array([target_func(*list(case)) for case in cases])
    fits = np.empty(len(pop))
    for i, org in enumerate(pop):
        y_actual = []
        for case in cases:
            # Evaluate the organism
            l = Linear([[0]+list(case)+[0], np.ravel(org)])
            l.run(kwargs['timeout'])
            y_actual = np.append(y_actual, l.regs[-1])
        # Calculate MSE
        fits[i] = sum((abs(y_target - y_actual)) ** 2) / len(cases)
        # Calculate RMSE
        if rmse:
            fits[i] **= 0.5
    return fits


def lgp_self_rep_rmse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""
    # 2D array of input variables for each test case
    cases = cartesian_prod(*domains)
    y_target = np.array([target_func(*list(case)) for case in cases])
    fits = np.empty(len(pop))
    for i, org in enumerate(pop):
        y_actual = []
        for case in cases:
            # Evaluate the organism
            l = Linear([[0]+list(case)+[0], org[0]])
            l.run(kwargs['timeout'])
            y_actual = np.append(y_actual, l.regs[-1])
        # Calculate RMSE
        fits[i] = (sum((abs(y_target - y_actual)) ** 2) / len(cases)) ** 0.5
    return fits


# def _smlgp_nim(org0, org1, **kwargs):
#
#     #
#     a = Linear([[0,21,0],org0])
#     a.run(kwargs['timeout'])
#     da = a.mem[0][]
#
#     b = Linear([[0,0],org1])

def _check_sylver_coinage(n , played_values):
    
    # Invalid by the definition of the game
    if n <= 1:
        return False
    # No previous moves
    elif not played_values:
        return True

    # Dynamic programming to test representability
    reachable = [False] * (n + 1)
    reachable[0] = True

    for i in range(1, n + 1):
        for a in played_values:
            if i - a >= 0 and reachable[i - a]:
                reachable[i] = True
                break

    # Move is invalid if n is in the semigroup
    return not reachable[n]











def _smlgp_sylver_coinage(org0, org1, **kwargs):

    num_turns = kwargs['num_turns']

    played_numbers = [0] * (num_turns * 2)
    # played_numbers = [0] * (num_turn

    # Initialize organism a
    a = Linear([[0] + played_numbers, org0], valid_ops=kwargs['ops'])
    # Initialize organism b
    b = Linear([[0] + played_numbers, org1], valid_ops=kwargs['ops'])

    for turn in range(0, num_turns, 2):

        # Update memory of played values
        a.mem[0] = [0] + [0] + played_numbers
        a.regs = a.mem[0]

        # Run until timeout
        a.run(kwargs['timeout'])
        # Extract final value played by a
        a_played = a.mem[0][1]
        print(a_played)

        # Check if the value is valid and save it
        if _check_sylver_coinage(a_played, played_numbers):
            # played_numbers.append(a_played)
            played_numbers[turn] = a_played
        else:
            return 0, turn

        turn += 1

        # Update memory of played values
        b.mem[0] = [0] + [0] + played_numbers
        b.regs = b.mem[0]
        # Run until timeout
        b.run(kwargs['timeout'])
        # Extract final value played by a
        b_played = b.mem[0][1]
        print(b_played)

        # Check if the value is valid and save it
        if _check_sylver_coinage(b_played, played_numbers):
            # played_numbers.append(b_played)
            played_numbers[turn] = b_played
        else:
            return turn, 0

    return num_turns+1, num_turns+1


def smlgp_compete(pop, **kwargs):
    """Randomly compete each organism against another organism"""

    shuffle_map = np.arange(len(pop))
    kwargs['rng'].shuffle(shuffle_map)
    fits = np.empty(len(pop))

    for i in range(0, len(pop), 2):

        index0 = shuffle_map[i]
        index1 = shuffle_map[i+1]
        org0 = pop[index0]
        org1 = pop[index1]

        fit0, fit1 = _smlgp_sylver_coinage(org0, org1, **kwargs)

        fits[index0] = fit0
        fits[index1] = fit1

    return fits



#
# Target Functions
#

def x2(x): return 2 * x
def multiply(x0,x1): return x0 * x1
def power(x0,x1): return x0 ** x1


#
# Crossover Functions
#

def one_point_crossover(a, b, **kwargs):
    cut_a = kwargs['rng'].integers(0, len(a) + 1)
    cut_b_min = max(cut_a + len(b) - kwargs['max_len'], cut_a - len(a) + kwargs['min_len'])
    cut_b_max = min(cut_a + len(b) - kwargs['min_len'], cut_a - len(a) + kwargs['max_len'])
    cut_b = kwargs['rng'].integers(cut_b_min, cut_b_max + 1)
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    return new_a, new_b


def two_point_crossover(a, b, **kwargs):
    # Difference in lengths of the sections to be swapped
    # diff_diff_cuts = len(a) - len(b)
    # kwargs['min_len'] <= len(a) + diff_diff_cuts <= kwargs['max_len']
    # kwargs['min_len'] <= len(b) - diff_diff_cuts <= kwargs['max_len']
    diff_diff_cuts_min = max(kwargs['min_len'] - len(a), len(b) - kwargs['max_len'])
    diff_diff_cuts_max = min(kwargs['max_len'] - len(a), len(b) - kwargs['min_len'])
    diff_diff_cuts = kwargs['rng'].integers(diff_diff_cuts_min, diff_diff_cuts_max + 1)
    # The length of a cut cannot be negative
    # 0 <= diff_cuts_a <= len(a)
    # 0 <= diff_cuts_a + diff_diff_cuts <= len(b)
    diff_cuts_a = kwargs['rng'].integers(max(0, -diff_diff_cuts), min(len(a), len(b) - diff_diff_cuts) + 1)
    diff_cuts_b = diff_cuts_a + diff_diff_cuts
    cut_a_0 = kwargs['rng'].integers(0, len(a) - diff_cuts_a + 1)
    cut_b_0 = kwargs['rng'].integers(0, len(b) - diff_cuts_b + 1)
    cut_a_1 = cut_a_0 + diff_cuts_a
    cut_b_1 = cut_b_0 + diff_cuts_b
    # Swap the two sections
    new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
    new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
    assert kwargs['min_len'] <= len(new_a) <= kwargs['max_len']
    assert kwargs['min_len'] <= len(new_b) <= kwargs['max_len']
    return new_a, new_b


def self_crossover(a,b,**kwargs):
    """Use the organisms to replicate and mutate self"""
    a_vars, a_solv, a_repl = a
    b_vars, b_solv, b_repl = b
    # Code used to replicate each sub block of code
    new_a_solv = [a_vars, a_repl, a_solv, b_solv]
    new_a_repl = [a_vars, a_repl, a_repl, b_repl]
    new_b_solv = [b_vars, b_repl, b_solv, a_solv]
    new_b_repl = [b_vars, b_repl, b_repl, a_repl]
    # Create and run a Linear runtime object
    new_a_solv = Linear(new_a_solv, rand=True).run(kwargs['timeout']).mem[3]
    new_a_repl = Linear(new_a_repl, rand=True).run(kwargs['timeout']).mem[3]
    new_b_solv = Linear(new_b_solv, rand=True).run(kwargs['timeout']).mem[3]
    new_b_repl = Linear(new_b_repl, rand=True).run(kwargs['timeout']).mem[3]
    # Build new organism from the replicated code
    new_a = [a_vars, new_a_solv, new_a_repl]
    new_b = [b_vars, new_b_solv, new_b_repl]
    return new_a, new_b

#
# Mutation Functions
#

# def point_mutation(code, **kwargs):
#     """Randomly change a value in a random line"""
#     # Duplicate the original
#     code = [line.copy() for line in code]
#     # Select a random line and sub line
#     index = kwargs['rng'].integers(len(code))
#     sub_index = kwargs['rng'].integers(4)
#     # Replace the argument
#     code[index][sub_index] = _random_line(**kwargs)[sub_index]
#     return code

def point_mutation(code, **kwargs):
    """Randomly change a value in a random line"""
    # Duplicate the original
    code_copy = code.copy()
    # Select a random line and sub line
    index = kwargs['rng'].integers(len(code))
    # Replace the argument
    code[index] = kwargs['rng'].integers(kwargs['max_value'])
    return code


#
# Debug
#

if __name__ == '__main__':
    pass

    org =  [8,  4,  3,  0 ,
11,  0,  0, 13 ,
 4, 13, 10,  1 ,
 4, 15,  1,  8 ,
 9,  1, 12, 13 ,
 3, 13,  9,  1 ,
14, 13,  3, 13 ,
11, 11,  6,  3 ,
 4, 13,  4,  0 ,
 9, 14,  1, 10 ,
 4,  3, 244,  7,
13,  0, 13,  5 ,
 7,  4, 13, 10 ,
 8,  6, 13,  0 ,
 7, 10,  1,  1 ,
35, 13,  0, 11]


    a = _check_sylver_coinage(16, [7])

    print(a)

    # prevs = [
    #     [5],
    #     [5,4],
    #     [5,4,11],
    #     [5,4,11,6],
    #     [5,4,11,6,7],
    #     [5,4,11,6,7,2],
    #     [5,4,11,6,7,2,3],
    # ]
    #
    # ns = range(15)
    #
    # vss = []
    # for prev in prevs:
    #     vs = []
    #     for n in ns:
    #         s = _check_sylver_coinage(n, prev)
    #         vs.append(n * s)
    #     vss.append(vs)
    #
    # # prev = [5]
    # # n = 10
    # # a = _check_sylver_coinage(n, prev)
    #
    # vss = np.array(vss, int)
    #
    # print(vss)