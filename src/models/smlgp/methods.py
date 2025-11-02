"""
Genetic programming functions specifically for the evolution of linear models.
Linear code is represented as a 2D arrays and converted to a Linear objects when evaluating.
"""
import numpy as np

from model import Linear
from src.utils.utils import cartesian_prod


#
# Initialization Functions
#

def _random_line(**kwargs):
    """Helper function to generating a single line of code"""
    return [
        kwargs['rng'].choice(kwargs['ops']),
        kwargs['rng'].integers(kwargs['max_len']),
        kwargs['rng'].integers(kwargs['max_value']),
        kwargs['rng'].integers(1+2*len(kwargs['mem_lens'])),
    ]

def random_code(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    code = [_random_line(**kwargs) for _ in range(init_len)]
    code = [sum(code, [])]
    return code

def random_2d_code(**kwargs):
    code = []
    for i in range(len(kwargs['init_max_lens'])):
        init_min_len = kwargs['init_min_lens'][i]
        init_max_len = kwargs['init_min_lens'][i]
        code.append(random_code(init_min_len=init_min_len, init_max_len=init_max_len, **kwargs))
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


# def unstable_self_rep(pop, **kwargs):
#     """Calculate the fitness value of all individuals in a population"""
#     fits = np.empty(len(pop))
#     for i,code in enumerate(pop):
#         code_1d = np.ravel(code)
#         l0 = run_self_rep(code, **kwargs)
#         out_code = np.array(l0.out).reshape(-1,4).tolist()
#         out_code_1d = np.ravel(out_code)
#         l1 = run_self_rep(out_code, **kwargs)
#
#         fit = sum((code_1d == l0.out) & (code_1d != 0))
#         fit += sum((out_code_1d == l1.out) & (out_code_1d != 0))
#         if (code_1d == out_code_1d).all():
#             fit = 0
#         fits[i] = fit
#     return fits


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

def point_mutation(code, **kwargs):
    """Randomly change a value in a random line"""
    # Duplicate the original
    code = [line.copy() for line in code]
    # Select a random line and sub line
    index = kwargs['rng'].integers(len(code))
    sub_index = kwargs['rng'].integers(4)
    # Replace the argument
    code[index][sub_index] = _random_line(**kwargs)[sub_index]
    return code


#
# Debug
#

if __name__ == '__main__':
    pass