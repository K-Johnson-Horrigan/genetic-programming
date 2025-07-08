"""
Genetic programming functions specifically for the evolution of linear models.
Linear code is represented as a 2D arrays and converted to a Linear objects when evaluating.
"""
import numpy as np

from src.genetics.classes.linear import Linear


#
# Initialization Functions
#

def _random_line(**kwargs):
    """Helper function for generating only a single line"""
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

def lgp_mse(pop, target_func, domains, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""

    # 2D array of input variables and each test case
    xs = [np.linspace(*domain) for domain in domains]
    xs = np.array(np.meshgrid(*xs)).reshape((len(xs), -1)).T

    y_target = np.array([target_func(*list(x)) for x in xs])

    fits = np.empty(len(pop))

    for i,code in enumerate(pop):
        # Pass all test cases as a single numpy array so that a semantic vector can be formed if needed

        y_actual = []

        for case in xs:
            l = Linear(code, num_reg=2)

            l.mem[1] = case[0]

            l.run(100)

            y_actual.append(l.mem[2])

        y_actual = np.array(y_actual)

        fit = (sum((abs(y_target - y_actual)) ** 2) / len(xs)) ** (1/2)
        fits[i] = fit

    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits


def self_rep(pop, **kwargs):
    """Calculate the fitness value of all individuals in a population against the target function for the provided domain"""

    fits = np.empty(len(pop))

    for i,code in enumerate(pop):

        code_1d = np.ravel(code)

        l = Linear(code, num_output_regs=len(code_1d))
        l.run(kwargs['timeout'])
        result = l.mem[1:len(code_1d)+1]
        diffs = abs(code_1d - result)
        fit = sum(diffs)
        fits[i] = fit

    fits = np.nan_to_num(fits, nan=np.inf, posinf=np.inf)
    return fits

#
# Target Functions
#

def x2(x): return 2 * x


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
    cut_a = kwargs['rng'].integers(0, len(a))
    cut_b = kwargs['rng'].integers(0, len(b))
    new_a = a[:cut_a] + b[cut_b:]
    new_b = b[:cut_b] + a[cut_a:]
    return new_a, new_b


def two_point_crossover(a, b, **kwargs):
    cut_a_0 = kwargs['rng'].integers(0, len(a))
    cut_b_0 = kwargs['rng'].integers(0, len(b))
    cut_a_1 = kwargs['rng'].integers(cut_a_0, len(a))
    cut_b_1 = kwargs['rng'].integers(cut_b_0, len(b))
    new_a = a[:cut_a_0] + b[cut_b_0:cut_b_1] + a[cut_a_1:]
    new_b = b[:cut_b_0] + a[cut_a_0:cut_a_1] + b[cut_b_1:]
    return new_a, new_b

#
# Debug
#

if __name__ == '__main__':
    # pass

    # # a,b,c,d = 1,2,3,4
    # x,y = 1,2
    #
    # code = [
    #     [Linear.STORE, y, x, Linear.DIRECT],
    #     [Linear.MUL,   y, 2, Linear.IMMEDIATE],
    #     [Linear.STOP,  0, 0, Linear.IMMEDIATE],
    # ]
    #
    # f = lgp_mse([code], target_func=x2, domains=[[1,4,3]])
    #
    # print(f)
    #
    # l = Linear(code, num_reg=2)
    # l.mem[x] = 3
    # l.run(100)
    # print(l.mem[y])

    # pc,a,b,t = 0,1,2,3
    # code = [
    #     [Linear.LOAD,  a, pc, Linear.DIRECT], # Copy PC to value of a-pointer
    #     [Linear.LOAD,  t,  a, Linear.INDIRECT], # Copy a-pointer to temp
    #     [Linear.STORE, t,  b, Linear.INDIRECT], # Copy temp to b-pointer
    #     [Linear.ADD,   a,  1, Linear.IMMEDIATE], # move a-pointer to next value
    #     [Linear.ADD,   b,  1, Linear.IMMEDIATE], # Move b-pointer to next value
    #     [Linear.IFEQ,  b, 32, Linear.IMMEDIATE], # b is at the end
    #     [Linear.STOP, 0, 0, Linear.IMMEDIATE], # Stop
    #     [Linear.SUB,  pc, 4*7, Linear.IMMEDIATE], # Return to start
    # ]

    pc, a, b, t = 0, 1, 2, 3
    code = [
        [Linear.ADD, pc, 4, Linear.IMMEDIATE], # Skip past next line
        [0,0,0,0], # Local variable storage
        [Linear.LOAD,  a, pc, Linear.DIRECT], # Copy PC to value of a-pointer
        [Linear.LOAD,  t,  a, Linear.INDIRECT], # Copy a-pointer to temp
        [Linear.STORE, t,  b, Linear.INDIRECT], # Copy temp to b-pointer
        [Linear.ADD,   a,  1, Linear.IMMEDIATE], # move a-pointer to next value
        [Linear.ADD,   b,  1, Linear.IMMEDIATE], # Move b-pointer to next value
        [Linear.IFEQ,  b, 32, Linear.IMMEDIATE], # b is at the end
        [Linear.STOP, 0, 0, Linear.IMMEDIATE], # Stop
        [Linear.SUB,  pc, 4*7, Linear.IMMEDIATE], # Return to start
    ]

    fits = self_rep([code], timeout=100)
    print(fits)

