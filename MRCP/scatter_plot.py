from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
from nash_solver.gambit_tools import load_pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

idx = 3
round = '10'
game_type = "szs"
MSS = "FP"
# root_path = './data/' + game_type + '_' + MSS + '/' + game_type + '_' + MSS + str(idx) + '/'

root_path = './data/10,4Blotto/'

regret_path = root_path + '10,4-Blotto_90335regret_of_samples_' + round + '.pkl'
improvement_path = root_path + '10,4-Blotto_90335performance_improvement_' + round + '.pkl'

regret = load_pkl(regret_path)
improvement = load_pkl(improvement_path)

NE_regret = 0.13
NE_improvement = 0.0048
MRCP_regret = 0.055
MRCP_improvement = 0.00013


plt.scatter(regret, improvement)
plt.plot(NE_regret, NE_improvement, '-ro')
# plt.plot(MRCP_regret, MRCP_improvement, '-go')

plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Regret of Training Target Profile', fontsize = 22)
plt.ylabel('Regret of MRCP Improvement', fontsize = 19)
plt.show()