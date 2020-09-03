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


DO_MRCP = genfromtxt('../data/MRCP_NE/DO_MRCP.csv', delimiter=',')
DO_NE = genfromtxt('../data/MRCP_NE/DO_NE.csv', delimiter=',')
FIC_MRCP = genfromtxt('../data/MRCP_NE/FIC_MRCP.csv', delimiter=',')
FIC_NE = genfromtxt('../data/MRCP_NE/FIC_NE.csv', delimiter=',')

DO_MRCP_std = genfromtxt('../data/MRCP_NE/DO_MRCP_std.csv', delimiter=',')
DO_NE_std = genfromtxt('../data/MRCP_NE/DO_NE_std.csv', delimiter=',')
FIC_MRCP_std = genfromtxt('../data/MRCP_NE/FP_MRCP_std.csv', delimiter=',')
FIC_NE_std = genfromtxt('../data/MRCP_NE/FP_NE_std.csv', delimiter=',')


# axes = plt.gca()
# axes.set_ylim([0,3])

X = np.arange(1, 101)


plt.plot(X, DO_NE, color="C1", label='DO w. NE-based regret')
plt.fill_between(X, DO_NE+DO_NE_std, DO_NE-DO_NE_std, alpha=0.1, color="C1")

plt.plot(X, FIC_NE, color="C2", label='FP w. NE-based regret')
plt.fill_between(X, FIC_NE+FIC_NE_std, FIC_NE-FIC_NE_std, alpha=0.1, color="C2")

plt.plot(X, DO_MRCP, color="C3", label='DO w. MRCP-based regret')
plt.fill_between(X, DO_MRCP+DO_MRCP_std, DO_MRCP-DO_MRCP_std, alpha=0.1, color="C1")

plt.plot(X, FIC_MRCP, color="C4", label='FP w. MRCP-based regret')
plt.fill_between(X, FIC_MRCP+FIC_MRCP_std, FIC_MRCP-FIC_MRCP_std, alpha=0.1, color="C1")



plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()

