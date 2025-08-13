"""File for testing ideas"""

from main_lgp import *

kwargs['rng'] = np.random.default_rng()

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


f = random_random_code(**kwargs)

f = [[0,0,0,0], *f, []]
f = Linear(f)

print(f)


