import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
from utils import *
from itertools import product
# from meta_strategies import double_oracle,fictitious_play

meta_games = [np.random.rand(10, 10), np.random.rand(10, 10)]
empirical_games = [[1,2,3,6], [3,5,7,9]]

caches = [Cache(), Cache()]
caches = find_all_deviation_payoffs(empirical_games=empirical_games,
                                    meta_game=meta_games,
                                    caches=caches)


prob_var = np.array([0.1, 0.3, 0.4, 0.2, 0.1, 0.3, 0.4, 0.2])
regret = upper_bouned_regret_of_variable(prob_var, empirical_games, meta_games, caches)

print(regret)

