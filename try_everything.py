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
from MRCP.regret_analysis import extend_prob, uniform_simplex_sampling

meta_games = [np.random.rand(10, 10), np.random.rand(10, 10)]
empirical_games = [[1,2,3,6], [3,5,7,9]]
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


rand_str = []
for player in range(2):
    num_strategies_in_EG = 4
    rand_str.append(uniform_simplex_sampling(num_strategies_in_EG))

print(rand_str)
strategies = extend_prob(rand_str, empirical_games, meta_games)
print(strategies)