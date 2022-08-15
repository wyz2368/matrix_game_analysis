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

# fic_mean = genfromtxt('./data/crd_tuning/'+ game + '/FP_mean.csv', delimiter=',')[8:21]
# DO_mean = genfromtxt('./data/crd_tuning/'+ game + '/DO_mean.csv', delimiter=',')[8:21]
# CRD_mean = genfromtxt('./data/crd_tuning/'+ game + '/CRD_mean.csv', delimiter=',')[8:21]

# fic_mean = genfromtxt('./data/real_world/hex_fic.csv', delimiter=',')
# DO_mean = genfromtxt('./data/real_world/hex_DO.csv', delimiter=',')
# CRD_mean = genfromtxt('./data/real_world/hex_crd.csv', delimiter=',')


DO_mean = np.array([2, 2, 2,2, 1.77777778, 1.77777778, 1.41927625, 1.24072417,
 1.03423594, 0.99277195, 0.89325864, 0.5020604])
fic_mean = np.array([ 2, 1.66666667, 1.99999999, 1.33225865, 1.05697859 ,1.05697859, 1.05697859, 0.94433695,
 0.78710142, 0.6187167 , 0.6187167 , 0.6187167])
CRD_mean = np.array([ 2.  ,1.66666667,
 1.845, 1.11381065, 1.33333334, 0.80093104, 0.88492352 ,0.63295092,
 0.43962214 ,0.47935835 ,0.46732767, 0.14947377])


# fic_std = genfromtxt('./data/crd_tuning/'+ game + '/FP_std.csv', delimiter=',')
# DO_std = genfromtxt('./data/crd_tuning/'+ game + '/DO_std.csv', delimiter=',')
# CRD_std = genfromtxt('./data/crd_tuning/'+ game + '/CRD_std.csv', delimiter=',')

# axes = plt.gca()
# axes.set_ylim([-0.1,1.25])
#
# fic_mean[0] = 1.9
# DO_mean[0] = 1.9
# CRD_mean[0] = 1.9


X = np.arange(1, len(CRD_mean)+1).astype(dtype=np.str)

plt.plot(X, fic_mean, color="C0", marker='o', label='FP')
# plt.fill_between(X, fic_mean+fic_std, fic_mean-fic_std, alpha=0.1, color="C0")

plt.plot(X, DO_mean, color="C2", marker='o', label='DO')
# plt.fill_between(X, DO_mean+DO_std, DO_mean-DO_std, alpha=0.1, color="C2")

plt.plot(X, CRD_mean, color="C1", marker='o', label='RRD')
# plt.fill_between(X, CRD_mean+CRD_std, CRD_mean-CRD_std, alpha=0.1, color="C1")


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 22)


plt.legend(loc="best", prop={'size': 22})

plt.tight_layout()

plt.show()

