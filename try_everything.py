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

# def project_onto_unit_simplex1(prob):
#     """
#     Project an n-dim vector prob to the simplex Dn s.t.
#     Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
#     :param prob: a numpy array. Each element is a probability.
#     :return: projected probability
#     """
#     prob_length = len(prob)
#     bget = False
#     sorted_prob = -np.sort(-prob)
#     tmpsum = 0
#
#     for i in range(1, prob_length):
#         print(i)
#         tmpsum = tmpsum + sorted_prob[i-1]
#         tmax = (tmpsum - 1) / i
#         if tmax >= sorted_prob[i]:
#             bget = True
#             break
#
#     if not bget:
#         tmax = (tmpsum + sorted_prob[prob_length-1] - 1) / prob_length
#
#     return np.maximum(0, prob - tmax)
#
# a = np.ones(5)
# variables = np.array([1.5, 1.5, 2, 0.5])
# pointer = 0
# sections = [2, 2]
# for ele in np.cumsum(sections):
#     variables[pointer:ele] = project_onto_unit_simplex1(variables[pointer:ele])
#     pointer = ele
#
# print(variables)

print(np.linalg.norm(np.array([2,2,1])))

