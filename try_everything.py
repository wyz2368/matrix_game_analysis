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
from scipy.stats import entropy
from MRCP.regret_analysis import extend_prob, uniform_simplex_sampling, sampling_scheme, profile_regret
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator

# meta_games = [np.random.rand(10, 10), np.random.rand(10, 10)]
# empirical_games = [[1,2,3,6], [3,5,7,9]]
meta_games = [np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]]), np.array([[1,2,3],
                        [4,5,6],
                        [7,8,9]])]

exact_calculator = minimum_regret_profile_calculator(full_game=meta_games)

empirical_games = [[1], [1]]
mrcp, regret = exact_calculator(empirical_games)
print(mrcp, regret)

empirical_games[0].append(2)
empirical_games[1].append(2)
mrcp, regret = exact_calculator(empirical_games)
print(mrcp, regret)

# empirical_games = [[2,0], [2,0]]
# mrcp, regret = exact_calculator(empirical_games)
# print(mrcp, regret)