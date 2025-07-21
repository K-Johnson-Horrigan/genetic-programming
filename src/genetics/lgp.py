"""
Genetic programming functions specifically for the evolution of linear models.
Linear code is represented as a 2D arrays and converted to a Linear objects when evaluating.
"""
import numpy as np

from src.genetics.classes.linear import Linear
from src.utils.utils import cartesian_prod


#
# Initialization Functions
#

def _random_line(**kwargs):
    """Helper function for generating a single line"""
    return [
        kwargs['rng'].choice(kwargs['ops']),
        kwargs['rng'].integers(kwargs['max_len']),
        kwargs['rng'].integers(kwargs['max_value']),
        kwargs['rng'].choice(kwargs['addr_modes']),
    ]

def random_code(**kwargs):
    """Generate a random list of transitions"""
    init_len = kwargs['rng'].integers(kwargs['init_min_len'], kwargs['init_max_len']+1)
    trans = [_random_line(**kwargs) for _ in range(init_len)]
    return trans


#
# Fitness Functions
#

def lgp_rmse(pop, target_func, domains, **kwargs):
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
            y_actual = np.append(y_actual, l.vars[-1])
        # Calculate MSE
        fits[i] = (sum((abs(y_target - y_actual)) ** 2) / len(cases)) ** 0.5
    return fits


def lgp_mse(pop, target_func, domains, **kwargs):
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
            y_actual = np.append(y_actual, l.vars[-1])
        # Calculate MSE
        fits[i] = sum((abs(y_target - y_actual)) ** 2) / len(cases)
    return fits


def lgp_error(pop, target_func, domains, **kwargs):
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
            y_actual = np.append(y_actual, l.vars[-1])
        # Calculate error
        fits[i] = sum(abs(y_target - y_actual))
    return fits


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

#
# Target Functions
#

def x2(x): return 2 * x
def multiply(x0,x1): return x0 * x1

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


# def test_all_two_point_crossover(a, b, **kwargs):
#     l = []
#     for cut_a_0 in range(len(a)+1):
#         for cut_a_1 in range(cut_a_0, len(a)+1):
#             for cut_b_0 in range(len(b)+1):
#                 for cut_b_1 in range(cut_b_0, len(b)+1):
#                     new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
#                     new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
#                     if (kwargs['min_len'] <= len(new_a) <= kwargs['max_len']) and (kwargs['min_len'] <= len(new_b) <= kwargs['max_len']):
#                         l.append(f'{new_a} {new_b}')
#     return l

#
# Debug
#

if __name__ == '__main__':
    pass

    # a = 'abcde'
    # b = '1234567'
    #
    # c,d = two_point_crossover(a, b, min_len=4, max_len=16, rng=np.random.default_rng())

    # min_len = 1
    # max_len = 40
    #
    # # al = test_all_two_point_crossover(a, b, min_len=min_len, max_len=max_len)
    # # al = list(set(al))
    # #
    # # l = calc_all_two_point_crossover(a, b, min_len=min_len, max_len=max_len)
    # # l = list(set(l))
    # #
    # # print(len(l), len(al))
    # # print(set(al)-set(l))
    #
    # # uc = np.array(np.unique(l, return_counts=True)).T
    # # print(uc)
    #
    # d = {}
    #
    # for _ in range(10000):
    #     ab = one_point_crossover(a, b, rng=np.random.default_rng(), min_len=0, max_len=40)
    #     if ab in d:
    #         d[ab] += 1
    #     else:
    #         d[ab] = 1
    #
    # for key in d:
    #     print(f'{key} {d[key]}')
    #
    # print(len(d))


    # ALMOST SELF REP
    # PROGRAM MEMORY
    #  0 │ SUB   │  1 │  3 │ IMMEDIATE
    #  4 │ LOAD  │  2 │  4 │ MEM_INDIRECT
    #  8 │ STORE │  2 │ 10 │ OUT_INDIRECT
    # 12 │ IFEQ  │  1 │  8 │ MEM_INDIRECT     THIS LINE HERE
    # 16 │ SUB   │  1 │  3 │ OUT_DIRECT
    # OUTPUT MEMORY
    #  0 │ STOP  │  1 │  3 │ IMMEDIATE
    #  4 │ LOAD  │  2 │  0 │ MEM_INDIRECT
    #  8 │ STORE │  0 │ 10 │ OUT_INDIRECT
    # 12 │ IFEQ  │  1 │  8 │ MEM_INDIRECT
    # 16 │ SUB   │  1 │  3 │ OUT_DIRECT
    # OUTPUT MEMORY
    #  0 │ STOP  │  0 │  0 │ IMMEDIATE
    #  4 │ STOP  │  0 │  0 │ IMMEDIATE
    #  8 │ STOP  │  0 │  0 │ IMMEDIATE
    # 12 │ STOP  │  0 │  0 │ IMMEDIATE
    # 16 │ STOP  │  0 │  0 │ IMMEDIATE

    # code = [
    #     [Linear.SUB   ,  1 ,  3 , Linear.IMMEDIATE],
    #     [Linear.LOAD  ,  2 ,  4 , Linear.MEM_INDIRECT],
    #     [Linear.STORE ,  2 , 10 , Linear.OUT_INDIRECT],
    #     # [Linear.IFEQ  ,  1 ,  8 , Linear.MEM_INDIRECT],
    #     [Linear.SUB   ,  1 ,  3 , Linear.OUT_DIRECT],
    # ]

#     PROGRAM
#     MEMORY
#     0 │ STORE │  2 │  9 │ OUT_INDIRECT
#     4 │ SUB   │  2 │  1 │ MEM_DIRECT
#     8 │ LOAD  │  1 │  5 │ MEM_INDIRECT
# 12 │ STORE │  1 │  8 │ OUT_INDIRECT
# OUTPUT
# MEMORY
# 0 │ STORE │  2 │  9 │ OUT_INDIRECT
# 4 │ ADD   │  2 │  1 │ MEM_DIRECT
# 8 │ LOAD  │  1 │  5 │ MEM_INDIRECT
# 12 │ STORE │  1 │  8 │ OUT_INDIRECT

    # code = [
    #     [Linear.STORE, 2, 9, Linear.OUT_INDIRECT],
    #     [Linear.SUB, 2, 1, Linear.MEM_DIRECT],
    #     [Linear.LOAD, 1, 5, Linear.MEM_INDIRECT],
    #     [Linear.STORE, 1, 8, Linear.OUT_INDIRECT],
    # ]

    # code = [
    #     [2, 5, 9, 4],
    #     [4, 2, 1, 5],
    #     [1, 7, 5, 6],
    #     [2, 7, 8, 4],
    # ]
    #
    # code = [
    #     [Linear.STORE, 2, 9, Linear.OUT_INDIRECT],
    #     [Linear.SUB, 2, 1, Linear.MEM_DIRECT],
    #     [Linear.LOAD, 1, 5, Linear.MEM_INDIRECT],
    #     [Linear.STORE, 1, 8, Linear.OUT_INDIRECT],
    # ]
    #
    # ll = run_self_rep(code, timeout=64)
    # print(ll)
    #
    # code = np.array(ll.out).reshape(-1,4).tolist()
    # print(code)
    # ll = run_self_rep(code, timeout=64)
    # print(ll)
    #
    # code = np.array(ll.out).reshape(-1,4).tolist()
    # print(code)
    # ll = run_self_rep(code, timeout=64)
    # print(ll)

    # Self-Rep / Crossover / Mutation
    # code = [[
    #     0, # PC
    #     0, # Random value
    #     0, # Copy pointer
    #     0, # Temp
    # ],[
    #     Linear.RAND,  1,  1, Linear.VARS_DIRECT,   # Generate random value
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.LOAD,  3,  2, Linear.CODE_INDIRECT, # Load temp value from MEM2
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.STORE, 3,  2, Linear.MEM2_INDIRECT, # Store temp value into MEM2
    #     Linear.ADD,   2,  1, Linear.IMMEDIATE,     # Increment copy pointer
    # ],[
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    # ]]

    #
    # code = [
    #     Linear.ADD, 2, 1, Linear.VARS_DIRECT,
    #     Linear.ADD, 2, 1, Linear.VARS_DIRECT,
    #     Linear.STOP, 0, 0, 0,
    # ]
    #
    # # l = Linear([[0,2,5,0], code])
    # # l.run(2)
    # # print(l)
    # #
    #
    # r = lgp_mse([code], x2, [[0,1,2,3,4]], timeout=3)
    #
    # print(r)


    # Multiply
    code = [
        [Linear.IFEQ, 2,  4, Linear.CODE_DIRECT],
        [Linear.STOP, 2,  9, Linear.VARS_DIRECT],
        [Linear.SUB,  2, 15, Linear.CODE_DIRECT],
        [Linear.ADD,  3, 13, Linear.VARS_DIRECT],
    ]

    # code = [
    #     [5, 2,  4, 3],
    #     [0, 2,  9, 1],
    #     [4, 2, 15, 3],
    #     [3, 3, 13, 1],
    # ]

    case = [2,10]
    l = Linear([[0] + list(case) + [0], np.ravel(code)])
    l.run(64)
    y_actual = l.vars[-1]
    print(l)

    # fits = lgp_rmse([code], multiply, [list(range(5)),list(range(5))], timeout=64)
    # print(fits)
