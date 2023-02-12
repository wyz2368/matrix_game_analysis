import numpy as np
import collections
from meta_strategies import adaptive_play, fictitious_play
from utils import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

m = 100
k = 20

# meta_game0 = np.array([[1.,  0.,  0.],
#                        [0.,  2.,  0.],
#                        [0.,  0.,  3.]])
#
# meta_game1 = np.array([[1.,  0.,  0.],
#                        [0.,  2.,  0.],
#                        [0.,  0.,  3.]])

dim = 10
a = list(np.arange(dim))
b = np.zeros((dim, dim))
np.fill_diagonal(b, a)
meta_game0 = np.copy(b)
meta_game1 = np.copy(b)

meta_games = np.array([meta_game0, meta_game1])

empirical_game0 = list(np.random.randint(dim, size=m))
empirical_game1 = list(np.random.randint(dim, size=m))

# empirical_game0 = list(np.ones(m))
# empirical_game1 = list(np.ones(m))

empirical_games = [empirical_game0, empirical_game1]

curve = []
for i in range(200):
    dev_strs, nashconv, nash_payoffs = adaptive_play(meta_games, empirical_games)
    # dev_strs, nashconv, nash_payoffs = fictitious_play(meta_games, empirical_games)
    empirical_games[0].append(dev_strs[0])
    empirical_games[1].append(dev_strs[1])
    print(nash_payoffs)
    curve.append(nash_payoffs[0])

plt.plot(curve)
plt.show()


