import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
from utils import *
import copy
from itertools import product
# from meta_strategies import double_oracle,fictitious_play
from nash_solver.gambit_tools import load_pkl
from nash_solver.replicator_dynamics_solver import replicator_dynamics
from nash_solver.projected_replicator_dynamics import projected_replicator_dynamics
from scipy.stats import entropy
from MRCP.regret_analysis import extend_prob, uniform_simplex_sampling, sampling_scheme, profile_regret
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator
from real_world_games.nash_clustering import nash_clustering

# meta_games = [np.random.rand(10, 10), np.random.rand(10, 10)]
# empirical_games = [[1,2,3,6], [3,5,7,9]]
# meta_games = [np.array([[1,2,3],
#                         [4,5,6],
#                         [7,8,9]]), np.array([[1,2,3],
#                         [4,5,6],
#                         [7,8,9]])]

# root_path = './MRCP/data/'
# meta_games = load_pkl(root_path + "meta_games.pkl")
#
#
# exact_calculator = minimum_regret_profile_calculator(full_game=meta_games)
#
# empirical_games = [[12, 31, 56, 87, 111, 116, 122, 144], [4, 111, 144, 147, 158, 176, 189]]
# for _ in range(6):
#     exact_calculator.clear()
#     mrcp, regret = exact_calculator(empirical_games)
#     print(regret)

meta_games = [np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]]), -np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])]

ne = nash_clustering(meta_games)
print(ne)

# a = [1,2,3,4,5,6]
# b = [1,2]
# print(a[b])
