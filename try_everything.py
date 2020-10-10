import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
from utils import *
from itertools import product
# from meta_strategies import double_oracle,fictitious_play

full_game = [np.random.rand(4, 4), np.random.rand(4, 4)]
print("full game:", full_game)
empirical_games = [[0,2], [1,3]]


caches = [Cache(), Cache()]
caches = find_all_deviation_payoffs(empirical_games=empirical_games,
                                    meta_game=full_game,
                                    caches=caches)

for player in range(2):
    for i, str in enumerate(empirical_games[1 - player]):
        print("str", str)
        print(caches[player].get(str))


