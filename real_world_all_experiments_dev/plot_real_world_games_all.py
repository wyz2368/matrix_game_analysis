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

game = 'hex'

fic_mean = genfromtxt('./data/'+ game + '/FP_mean.csv', delimiter=',')
DO_mean = genfromtxt('./data/'+ game + '/DO_mean.csv', delimiter=',')
PRD_mean = genfromtxt('./data/'+ game + '/PRD_mean.csv', delimiter=',')
CRD_mean = genfromtxt('./data/'+ game + '/CRD_mean.csv', delimiter=',')


fic_std = genfromtxt('./data/crd_tuning/'+ game + '/FP_std.csv', delimiter=',')
DO_std = genfromtxt('./data/crd_tuning/'+ game + '/DO_std.csv', delimiter=',')
PRD_std = genfromtxt('./data/crd_tuning/'+ game + '/DO_std.csv', delimiter=',')
CRD_std = genfromtxt('./data/crd_tuning/'+ game + '/CRD_std.csv', delimiter=',')

# axes = plt.gca()
# axes.set_ylim([-0.1,1.25])
#
# fic_mean[0] = 1.9
# DO_mean[0] = 1.9
# CRD_mean[0] = 1.9


X = np.arange(1, len(CRD_mean)+1).astype(dtype=np.str)

plt.plot(X, fic_mean, color="C0", marker='o', label='FP')
plt.fill_between(X, fic_mean+fic_std, fic_mean-fic_std, alpha=0.1, color="C0")

plt.plot(X, DO_mean, color="C2", marker='o', label='DO')
plt.fill_between(X, DO_mean+DO_std, DO_mean-DO_std, alpha=0.1, color="C2")

plt.plot(X, PRD_mean, color="C3", marker='o', label='DO')
plt.fill_between(X, PRD_mean+PRD_std, PRD_mean-PRD_std, alpha=0.1, color="C3")

plt.plot(X, CRD_mean, color="C1", marker='o', label='RRD')
plt.fill_between(X, CRD_mean+CRD_std, CRD_mean-CRD_std, alpha=0.1, color="C1")


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 22)


plt.legend(loc="best", prop={'size': 22})

plt.tight_layout()

plt.show()

