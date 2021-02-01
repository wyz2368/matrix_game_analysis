from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 51, 2) # window size 51, polynomial order 3

window_size = 55
order = 2

########### Plot alpha rank filter ############

plt.figure()
# plt.title("NashConv Curves ", fontsize = 22)

game = "hex"

fic_MRCP = genfromtxt('./data/real_world/' + game + '_fic.csv', delimiter=',')
DO_MRCP = genfromtxt('./data/real_world/' + game + '_DO.csv', delimiter=',')
CRD_MRCP = genfromtxt('./data/real_world/' + game + '_crd.csv', delimiter=',')

# axes = plt.gca()
# axes.set_ylim([0,1.5])

X = np.arange(1, len(DO_MRCP)+1).astype(dtype=np.str)

plt.plot(X, fic_MRCP, color="C0", marker='o', label='FP')
plt.plot(X, DO_MRCP, color="C1", marker='o', label='DO')
plt.plot(X, CRD_MRCP, color="C2", marker='o', label='RRD')


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret of MRCP', fontsize = 19)

plt.legend(loc="best", prop={'size': 17})

plt.show()

