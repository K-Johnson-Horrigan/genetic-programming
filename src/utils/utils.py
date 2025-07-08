import numpy as np

def choice(arr, rng):
    """
    Return a random element of the given array without casting.
    This exists to simplify code.
    See: https://github.com/numpy/numpy/issues/10791
    """
    return arr[rng.choice(len(arr))]


def cartesian_prod(*args):
    """Returns the cartesian product of all arguments."""
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


def to_tuple(a):
    """Converts an ndarray into tuples."""
    if type(a) == np.ndarray:
        return tuple(to_tuple(i) for i in a)
    else:
        return int(a)