import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
# from utils import *
from itertools import product
# from meta_strategies import double_oracle,fictitious_play

# meta_game = np.array([[1,2,3],
#                       [4,5,6],
#                       [7,8,9]])
# print(np.shape(meta_game))
#
# meta_games = [meta_game, -meta_game]
#
# empirical_games = [[0,1,1], [0,1,1]]
#
# dev_strs, nashconv = double_oracle(meta_games, empirical_games, './')
# print(dev_strs)
# print(nashconv)
#
#
# dev_strs, nashconv = fictitious_play(meta_games, empirical_games, './')
# print(dev_strs)
# print(nashconv)

# def proj(a, y):
#     l = y / a
#     # print("l, y:", l, y)
#     idx = np.argsort(l)
#     d = len(l)
#
#     evalpL = lambda k: np.sum(a[idx[k:]] * (y[idx[k:]] - l[idx[k:]] * a[idx[k:]])) - 1
#
#     def bisectsearch():
#         idxL, idxH = 0, d - 1
#         L = evalpL(idxL)
#         H = evalpL(idxH)
#         # print("L, H:", L, H)
#
#         if L < 0:
#             return idxL
#
#         while (idxH - idxL) > 1:
#             iMid = int((idxL + idxH) / 2)
#             M = evalpL(iMid)
#
#             if M > 0:
#                 idxL, L = iMid, M
#             else:
#                 idxH, H = iMid, M
#
#         return idxH
#
#     k = bisectsearch()
#     lam = (np.sum(a[idx[k:]] * y[idx[k:]]) - 1) / np.sum(a[idx[k:]])
#
#     x = np.maximum(0, y - lam * a)
#
#     return x

def uniform_simplex_sampling(dim):
    """
    Uniform sample a unit simplex.
    :param dim: the dimension of sampled vector.
    :return:
    """
    vec = np.random.rand(dim+1)
    vec[0] = 0
    vec[-1] = 1
    vec = np.sort(vec)
    output = np.zeros(dim)
    for i in range(dim):
        output[i] = vec[i+1] - vec[i]

    return output

if 0:
    print("fads")

