"""File for testing ideas"""
from src.utils.save import load_kwargs

"""
saves
    [model]
        [experiment]
            data
                [test]
                    [seed]
                        fits.npy
                        pops.npy
            plots
            kwargs.json
src
    models
        __init__.py
        [model]
            __init__.py
            class.py
            methods.py
    utils
        plot.py
        save.py
        utils.py
    evolution
        evolution.py
"""






import numpy as np

from src.models.network.methods import *

kwargs = load_kwargs('test_0', '../saves/network/')
kwargs['interf'] = np.array(kwargs['interf'])

org = [1,1,1,1,1,1]

pop = [org]

pop = np.array(pop)

f = total_interference(pop, **kwargs)

print(f)











# from main_lgp import *

# kwargs['rng'] = np.random.default_rng()

# a = random_self_rep_code(**kwargs)
# b = random_self_rep_code(**kwargs)
#
# a[0] = [1] * len(a[0])
# b[0] = [2] * len(b[0])
#
# # a = Linear([[0,0,0,0], *a])
# # b = Linear([[0, 0, 0, 0], *b])
#
# a[0] = [
#     Linear.IFEQ, 2,  4, Linear.CODE_DIRECT,
#     Linear.STOP, 2,  9, Linear.VARS_DIRECT,
#     Linear.SUB,  2, 15, Linear.CODE_DIRECT,
#     Linear.ADD,  3, 13, Linear.VARS_DIRECT,
# ]
#
# fits = lgp_self_rep_rmse([a], **kwargs)
#
# print(fits)

# aa,bb = self_crossover(a,b,**kwargs)
#
# print(Linear([[0,0,0,0],*aa]))
# # print(bb)


# f = random_code(**kwargs)

# f = [[0,0,0,0], *f, []]
# f = Linear(f)

# print(f)


# b = np.zeros((3,3),int)

# b = np.zeros((3,),int)
#
#
#
#
# game = []
# plays = [0,3,0,3,0]
# # Map the moves to indices
# mapping = list(range(9))
# plays = [mapping.pop(i) for i in plays]
# # Create the board and play the moves
# board = np.zeros((9,),int)
# board[plays[::2]] = -1
# board[plays[1::2]] = 1
# board = board.reshape((3,3))
#
# print(board)
#
# # s = np.sum(board, axis=0)
# # ss = np.sum(board, axis=1)
# s = np.sum(board[(0,1,2),(0,1,2)])
# s = np.sum(board[(0,1,2),(2,1,0)])
#
# # s = np.sum(board, axis=0)
#
#
# print(s)










