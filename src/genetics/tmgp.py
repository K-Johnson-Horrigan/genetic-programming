"""
Genetic programming functions specifically for the evolution of Turing Machines.
Turing Machines are represented as 3D arrays and are converted to TM objects when evaluating.
"""

import numpy as np

from src.genetics.classes.tm import TM
from src.utils.utils import choice



#
# Initialization Functions
#

def random_trans_array(shape=None, rng=np.random.default_rng(), **kwargs):
    """Generate a random list of transitions"""
    shape = [len(kwargs['states']), len(kwargs['symbols'])] if shape is None else shape
    new_states = rng.choice(kwargs['states'], shape + [1])
    new_symbols = rng.choice(kwargs['symbols'], shape + [1])
    new_moves = rng.choice(kwargs['moves'], shape + [kwargs['tape_dim']])
    trans = np.concat((new_states, new_symbols, new_moves), axis=-1)
    return trans


#
# Fitness Functions
# Pattern Matching
#

def _array_diff(a, b):
    """Helper function to find the differences in two arrays"""
    a = np.array(a)
    b = np.array(b)
    a = np.trim_zeros(a)
    b = np.trim_zeros(b)
    b_pad = [(0,i) for i in np.maximum(0, np.array(a.shape) - b.shape)]
    a_pad = [(0,i) for i in np.maximum(0, np.array(b.shape) - a.shape)]
    a = np.pad(a, a_pad)
    b = np.pad(b, b_pad)
    diff = np.sum(abs(0 + a - b))
    return diff


def diff_zeros(pop, **kwargs):
    """Difference in zeros"""
    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tape = TM(trans).run(kwargs['tm_timeout'])
        fit = _array_diff(tape != 0, np.array(kwargs['target']) != 0)
        fits[i] = fit
    return fits


def diff_values(pop, **kwargs):
    """Difference in values"""
    fits = np.empty(len(pop))
    for i,trans in enumerate(pop):
        tape = TM(trans).run(kwargs['tm_timeout'])
        fit = _array_diff(tape, kwargs['target'])
        fits[i] = fit
    return fits


#
# Fitness Functions
# Maze Solving
#

def gen_maze(shape=None, rng=np.random.default_rng(), maze=None, pos=None, init_call=True):
    if init_call:
        maze = np.pad(np.ones(shape, int), 1)
        pos = (2,)*len(shape)
    # Set position to zero
    maze[pos] = 0
    # Cartesian product of [-2, 0, 2] repeated N times
    neighbors = np.array(np.meshgrid(*([[-2, 0, 2]] * len(shape)))).T.reshape(-1, len(shape))
    # Only keep rows with exactly one nonzero element
    neighbors = neighbors[np.sum(neighbors != 0, axis=1) == 1]
    # Advanced numpy indexing to get all neighbors
    neighbors = tuple((neighbors + [pos]).T)
    while True:
        # List of all valid new position
        options = np.array(neighbors).T[maze[neighbors] == 1]
        if len(options) == 0:
            # Backtrack if there are no options
            # Trim the edges of the maze if this is the initial function call
            if init_call:
                maze = maze[1:-1, 1:-1]
            return maze
        else:
            # Select new location
            new_pos = choice(options, rng)
            # Create a path from the current position to the new position
            maze[tuple((new_pos + pos) // 2)] = 0
            # Recursively call at the new point
            gen_maze(shape, rng, maze, tuple(new_pos), False)


def gen_spiral_maze(shape=None):
    # Mask determines that only one axis changes at a time and that they alternate
    mask = ((1,0), (0,1), (-1,0), (0,-1))
    # Length of walls along the (Y,X) axis
    length = np.array(shape) - 1
    wall_start = np.array((0,0))
    step = 0
    maze = np.zeros(shape, int)
    # Repeat while there are positive wall lengths
    while (length>0).any():
        wall_end = wall_start + length * mask[step%4]
        # Slice values must go from low (inclusive) to high (exclusive)
        maze_slice = np.array((np.minimum(wall_start, wall_end), np.maximum(wall_start, wall_end)+1))
        # Draw wall
        maze[maze_slice[0][0]:maze_slice[1][0], maze_slice[0][1]:maze_slice[1][1]] = 1
        # Decrease the wall length by 2 for the axis that was drawn
        length = length - 2 * np.abs(mask[step%4])
        step += 1
        # Next wall will start at the end of this wall
        wall_start = wall_end
    # Complete the boarder around the maze
    maze[0,1:] = 1
    maze[1,-1] = 1
    # Flip X axis to move maze start to (1,1)
    maze = maze[:,::-1]
    return maze


def format_maze(maze):
    """Make each cell 3x3 and assign values to each type of wall"""
    shape = maze.shape
    maze = maze.repeat(3, axis=0).repeat(3, axis=1)
    # Cartesian product of [-1,0,1] repeated N times
    shifts = np.array(np.meshgrid(*([[-1, 0, 1]] * len(shape)))).T.reshape(-1, len(shape))
    # Only keep rows with exactly one nonzero element
    shifts = shifts[np.sum(shifts != 0, axis=1) == 1]
    # Stack of all neighbors
    walls = 1 - np.array([np.roll(maze, shift, (0, 1)) for shift in shifts])
    # All walls missing exactly 0 neighbors
    core = np.sum(walls, axis=0)==0
    # All walls missing exactly 1 neighbor
    index = np.broadcast_to(np.sum(walls, axis=0)!=1, walls.shape)
    walls[index] = 0
    # Give a unique value for each wall depending on its neighbors
    walls = np.array(range(1, len(shifts)+1)).reshape(-1, 1, 1) * walls
    walls = np.sum(walls, axis=0)
    # Replace core walls with the largest value
    walls[core] = 5
    # Trim outside
    walls = walls[1:-1, 1:-1]
    return walls


def solve_maze(maze, start=(3, 3)):
    """Solves the maze by replacing each value with the distance from the end"""
    maze = np.array(maze)
    shape = maze.shape
    # Floor value is an identity element when used with minimum
    FLOOR_VALUE = np.iinfo(np.int32).max
    # Wall value is an absorbing element when used with minimum
    WALL_VALUE = -FLOOR_VALUE
    # Modify maze to use the special floor and wall values
    maze = np.array([FLOOR_VALUE,WALL_VALUE])[maze]
    # Set the initial value at the start of the maze and let it spread outwards
    maze[start] = 1
    # Repeat until there are no empty floor values
    while FLOOR_VALUE in maze:
        # Cartesian product of [-1,0,1] repeated N times
        shifts = np.array(np.meshgrid(*([[-1, 0, 1]] * len(shape)))).T.reshape(-1, len(shape))
        # Only keep rows with exactly one nonzero element
        shifts = shifts[np.sum(shifts != 0, axis=1) == 1]
        # Expand the current path in all directions and increase the value by 1
        # Taking the abs of the maze results in neither the walls or floors having any influence on the path
        paths = np.min([np.roll(abs(maze), shift, (0, 1)) for shift in shifts], axis=0) + 1
        # Replace floor values with a path, wall values cannot be changed
        maze = np.minimum(maze, paths)
    maze[maze==WALL_VALUE] = 0
    return maze


def _run_maze_tm(trans, **kwargs):
    """Setup and run a TM to solve a maze"""
    # trans = trans + [
    #     [state, ((-1, -1, -1), (-1, 1, -1), (-1, -1, -1)), 'halt', ((-1, -1, -1), (-1, 1, -1), (-1, -1, -1))]
    #     for state in kwargs['states']
    # ]
    tm = TM(trans)
    tm.write_tape(kwargs['target'])
    tm.head_pos = (3,3)
    tm.run(kwargs['tm_timeout'])
    return tm


def maze_fitness(pop, **kwargs):
    """Difference in values"""
    maze = np.array(kwargs['target'])
    sol = kwargs['maze_sol']
    fits = np.empty(len(pop))
    for i, trans in enumerate(pop):
        tape = _run_maze_tm(np.array(trans), **kwargs).get_tape_as_array()
        if tape.shape != sol.shape or (tape==maze).all():
            fit = 0
        else:
            # m = sol[tape!=maze]
            fit = np.max(sol[tape!=maze])
        fits[i] = fit
    return fits


#
# Mutation Functions
#

def macro_mutation(trans, rng=np.random.default_rng(), **kwargs):
    """Randomly change a value in a random transition"""
    trans = trans.copy()
    state_index = rng.choice(kwargs['states'])
    symbol_index = rng.choice(kwargs['symbols'])
    index = (state_index, symbol_index)
    trans[index] = random_trans_array(shape=[], rng=rng, **kwargs)
    return trans



#
# Crossover Functions
#

def flattened_one_point_crossover(parent_one, parent_two, rng=np.random.default_rng(), **kwargs):
    shape = parent_one.shape
    # Children start as flattened copies of parents
    child_one = parent_one.flatten()
    child_two = parent_two.flatten()
    # Point cannot be either end value
    point_one = rng.integers(1, len(child_one) - 1)
    point_two = None
    # Swap slices
    child_one[point_one:point_two] = parent_two.flatten()[point_one:point_two]
    child_two[point_one:point_two] = parent_one.flatten()[point_one:point_two]
    child_one = child_one.reshape(shape)
    child_two = child_one.reshape(shape)
    return child_one, child_two


def flattened_two_point_crossover(parent_one, parent_two, rng=np.random.default_rng(), **kwargs):
    shape = parent_one.shape
    # Children start as flattened copies of parents
    child_one = parent_one.flatten()
    child_two = parent_two.flatten()
    # Points must be at least 1 value apart
    point_one = rng.integers(0, len(child_one) - 1)
    point_two = rng.integers(point_one + 1, len(child_one))
    # Swap slices
    child_one[point_one:point_two] = parent_two.flatten()[point_one:point_two]
    child_two[point_one:point_two] = parent_one.flatten()[point_one:point_two]
    child_one = child_one.reshape(shape)
    child_two = child_one.reshape(shape)
    return child_one, child_two


def axis_one_point_crossover(parent_one, parent_two, rng=np.random.default_rng(), **kwargs):
    # Select the axis to slice over
    axis = np.random.randint(0, 3)
    # None is default as it slices the entire shape
    slices = [None] * 6
    # Slice cannot be either end value
    slices[2 * axis] = rng.integers(1, parent_one.shape[0] - 1)
    # Children start as copies of parents
    child_one = parent_one.copy()
    child_two = parent_two.copy()
    # Swap slices
    child_one[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]] = parent_two[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]]
    child_two[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]] = parent_one[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]]
    return child_one, child_two


def axis_two_point_crossover(parent_one, parent_two, rng=np.random.default_rng(), **kwargs):
    # Select the axis to slice over
    axis = np.random.randint(0, 3)
    # None is default as it slices the entire shape
    slices = [None] * 6
    # Slices must be at least 1 value apart
    slices[2 * axis] = rng.integers(0, parent_one.shape[0] - 1)
    slices[2 * axis + 1] = rng.integers(slices[2 * axis] + 1, parent_one.shape[0])
    # Children start as copies of parents
    child_one = parent_one.copy()
    child_two = parent_two.copy()
    # Swap slices
    child_one[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]] = parent_two[slices[0]:slices[1], slices[2]:slices[3], slices[4]:slices[5]]
    child_two[slices[0]:slices[1],slices[2]:slices[3],slices[4]:slices[5]] = parent_one[slices[0]:slices[1], slices[2]:slices[3], slices[4]:slices[5]]
    return child_one, child_two


def chunk_crossover(parent_one, parent_two, rng=np.random.default_rng(), **kwargs):
    shape = parent_one.shape
    # Default values are invalid so this functions as a do-while
    # TODO find a better implementation with the same probabilities
    x0 = y0 = z0 = 0
    x1, y1, z1 = shape
    # Create random chunk that is not the same as no change
    while x0 == y0 == z0 == 0 and (x1, y1, z1) == shape:
        x0 = rng.integers(0, parent_one.shape[0] - 1)
        x1 = rng.integers(x0 + 1, parent_one.shape[0])
        y0 = rng.integers(0, parent_one.shape[1] - 1)
        y1 = rng.integers(y0 + 1, parent_one.shape[1])
        z0 = rng.integers(0, parent_one.shape[2] - 1)
        z1 = rng.integers(z0 + 1, parent_one.shape[2])
        # print('yoink')
    # Children start as copies of parents
    child_one = parent_one.copy()
    child_two = parent_two.copy()
    # Swap values
    child_one[x0:x1, y0:y1, z0:z1] = parent_two[x0:x1, y0:y1, z0:z1]
    child_two[x0:x1, y0:y1, z0:z1] = parent_one[x0:x1, y0:y1, z0:z1]
    return child_one, child_two


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

    # t = random_trans_array(shape=[], states=[0,1], symbols=[2,3], moves=[-1,-2], tape_dim=2)
    #
    # print(t)

    # trans = np.empty((4,2,4), int)
    # trans[(0, 0)] = [1, 1,  1,  0]
    # trans[(1, 0)] = [2, 1,  0, -1]
    # trans[(2, 0)] = [3, 1, -1,  0]
    # trans[(3, 0)] = [0, 1,  0,  1]
    # trans[(0, 1)] = [3, 0, -1,  0]
    # trans[(1, 1)] = [0, 0,  0,  1]
    # trans[(2, 1)] = [1, 0,  1,  0]
    # trans[(3, 1)] = [2, 0,  0, -1]
    #
    # # print(trans)
    #
    # trans1 = macro_mutation(trans, states=[0,1,2,3], symbols=[0,1], moves=[-1,0,1], tape_dim=2)
    #
    # print(trans == trans1)


    # maze = gen_maze((9,9))
    # maze = _format_maze(maze)
    #
    # maze_sol = _solve_maze((maze!=0)*1, (3,3))
    #
    # plt.imshow(maze_sol)
    # plt.show()

    # t = np.zeros((2,2,3))
    #
    # print(t)
    #
    # print(t[0,1])

    # m = [
    #     [[],[]],
    # ]

    # maze = to_tuple(maze)
    # maze_sol =
    # X=-1
    # trans = [
    #     ['start', [[X,X,X],[X,0,X],[X,X,X]], 'start', [[X,X,X],[X,1,X],[X,X,X]], +1, 0],
    # ]
    # trans = [
    #     ['start', 0, 'start', 0, +1, 0]
    # ]
    # pop = [trans] * 100

    # t0 = time.time()
    # tape = _run_maze_tm(trans, tm_timeout=100, states=['start'], target=maze)
    # fits = maze_fitness(pop, tm_timeout=100, states=['start'], target=maze, maze_sol=maze_sol)
    # t1 = time.time()
    # total = t1-t0
    # print(fits)

    # times = np.array(TM.times)
    # print(sum(times))
    # print(times.mean())
