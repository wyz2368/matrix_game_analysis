import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
from utils import *
from itertools import product
# from meta_strategies import double_oracle,fictitious_play
from nash_solver.gambit_tools import load_pkl
from scipy.stats import entropy

# meta_games = [np.random.rand(10, 10), np.random.rand(10, 10)]
# empirical_games = [[1,2,3,6], [3,5,7,9]]
# probs = [np.random.rand(10), np.random.rand(10)]
# for prob in probs:
#     prob /= np.sum(prob)
# deviation_payoff_in_EG = deviation_within_EG(meta_games, empirical_games, probs)
#
# print(deviation_payoff_in_EG, probs)
#
# payoff_vec = benefitial_deviation_pure_strategy_profile(meta_games, opponent=1, strategy=5, base_value=deviation_payoff_in_EG)
# print(payoff_vec)
# print(np.random.choice(payoff_vec))

a = np.shape(np.random.rand(5, 5))
print(a[0])