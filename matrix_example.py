import numpy as np
from meta_strategies import fictitious_play, double_oracle

import os
from nash_solver.gambit_tools import load_pkl
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

meta_game = np.array([[0, -0.1, -3],
                      [0.1, 0, 2],
                      [3, -2, 0]])

meta_game2 = np.array([[0, 2, -3],
                      [-3, 0, 0],
                      [-4, 1, 5]])

meta_games = [meta_game, -meta_game2]

empirical_games = [[0], [0]]

deepmind_fic = []
do_fic = [6., 3., 1.4, 1.4, 1.4]
do = [6, 4, 0, 0, 0]
fic = [6, 4, 4, 0, 0]

for _ in range(5):
    dev_strs, nashconv = double_oracle(meta_games, empirical_games, "./real_world_all_experiments_dev")
    deepmind_fic.append(nashconv)
    for i, str in enumerate(dev_strs):
        empirical_games[i].append(str)

print("NashConv:", deepmind_fic)



# x = [1,2,3,4,5]
#
#
# plt.plot(x, fic, '-oC1', label= "NE-based regret of FP")
# plt.plot(x, do, '-oC2', label= "NE-based regret of DO")
# plt.plot(x, deepmind_fic, '-oC0', label= "uniform-based regret of FP")
# plt.plot(x, do_fic, '-oC3', label= "uniform-based regret of DO")
#
# plt.xlabel("Number of Iterations")
# plt.ylabel("Regret")
#
# plt.xticks(x)
# # plt.title("")
# plt.legend(loc="best")
# plt.show()