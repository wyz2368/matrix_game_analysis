import os
from nash_solver.gambit_tools import load_pkl
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import savgol_filter
import math

load_path = os.getcwd() + '/data/data1/'
zero_sum_DO = load_pkl(load_path + 'zero_sum_DO.pkl')
zero_sum_FP = load_pkl(load_path + 'zero_sum_FP.pkl')
zero_sum_DO_FP = load_pkl(load_path + 'zero_sum_DO_SP.pkl')

zero_sum_DO = np.mean(zero_sum_DO, axis=0)
zero_sum_FP = np.mean(zero_sum_FP, axis=0)
zero_sum_DO_FP = np.mean(zero_sum_DO_FP, axis=0)

# idx = 6
# zero_sum_DO = zero_sum_DO[idx]
# zero_sum_FP = zero_sum_FP[idx]
# zero_sum_DO_FP = zero_sum_DO_FP[idx]

window_size = 15
order = 2

# Focus on fictitious play
fic_zero_sum_DO_FP = []
for i in range(len(zero_sum_DO_FP)):
    if i % 2 == 0:
        fic_zero_sum_DO_FP.append(zero_sum_DO_FP[i])

fic_zero_sum_DO_FP = savgol_filter(fic_zero_sum_DO_FP, window_size, order)
zero_sum_DO = savgol_filter(zero_sum_DO, window_size, order)
zero_sum_FP = savgol_filter(zero_sum_FP, window_size, order)


print(len(fic_zero_sum_DO_FP), len(zero_sum_DO))



x = np.arange(1, len(zero_sum_DO)+1)
plt.plot(x, zero_sum_DO, '-C2', label= "NE-based regret of DO")
plt.plot(x, zero_sum_FP, '-C0', label= "uniform-based regret of FP")
plt.plot(np.arange(1, 2*len(fic_zero_sum_DO_FP)+1, 2), fic_zero_sum_DO_FP, '-C1', label= "NE-based of FP")
# plt.plot(x, zero_sum_DO_FP, '-C1', label= "DO+FP")



plt.xlabel("Number of Iterations")
plt.ylabel("Regret")
# plt.title("Average NashConv over 30 runs in Synthetic Zero-Sum Game")
plt.legend(loc="best")
plt.show()

