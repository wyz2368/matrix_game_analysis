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

game = 'hex'

# fic_mean = genfromtxt('./data/crd_tuning/'+ game + '/FP_mean.csv', delimiter=',')[8:21]
# DO_mean = genfromtxt('./data/crd_tuning/'+ game + '/DO_mean.csv', delimiter=',')[8:21]
# CRD_mean = genfromtxt('./data/crd_tuning/'+ game + '/CRD_mean.csv', delimiter=',')[8:21]

# fic_mean = genfromtxt('./data/real_world/hex_fic.csv', delimiter=',')
# DO_mean = genfromtxt('./data/real_world/hex_DO.csv', delimiter=',')
# CRD_mean = genfromtxt('./data/real_world/hex_crd.csv', delimiter=',')


fic_mean = np.array([293.14333333, 167.37666667, 135.051513, 105.07925926, 85.5160581, 74.876875, 64.67253333,
  55.61157407, 52.04836735, 44.81869792])
fic_std = np.array([164.00325898, 45.47756362, 25.80414434, 21.47363016, 9.23702408,
   6.22534568, 9.04036648, 4.6660138, 4.50569644, 4.34898299])


DO_mean = np.array([2.93143333e+02, 1.45009415e+02, 103,  8.22188984e+01, 53, 2.32088912e+01,
 9.24386219e-01, 3.65183084e-01, 8.65183084e-04, 8.65183084e-04])
DO_std = np.array([1.64003259e+02, 6.59372515e+01, 9.10677619e+01, 3.28212225e+01,
 4.57645129e+01, 1.59665865e+01, 1.59665865e-04, 1.59665865e-04,
 1.59665865e-04, 1.59665865e-04])

PRD_mean = np.array([293.14333333,  140.70118593,  100.435513,  60.47232372,  45.47232372,
  25.47232372,  25.47232372,  25.47232372,  25.47232372,  25.47232372])
PRD_std = np.array([97.20859713, 16.03246177, 12.59900503, 8.80064306, 8.80064306, 8.80064306,
 8.80064306, 8.80064306, 8.80064306, 8.80064306])


CRD_mean = np.array([293.14333333, 109.87447973,  10.66621253, 2.34603175, 0.19745766, 2.65183084e-04, 2.65183084e-04, 8.65183084e-04, 0, 0])
CRD_std = np.array([1.64003259e+02, 7.06553277e+01, 1.28735695e+01, 2.61079727e+00,
 1.37940082e-01, 2.10563716e-01, 1.89165163e-01, 2.71518733e-01,
 3.03467965e-01, 1.04041673e-02])

DO_SWRO_plus = np.array([293.14333333,  5.16,      0. ,          0. ,          0.,
   0.   ,        0.      ,     0.       ,    0.      ,     0.])
DO_SWRO_plus_std = np.array([52.8,  7.29,      0. ,          0. ,          0.,
   0.   ,        0.      ,     0.       ,    0.      ,     0.])

DO_SWRO_minus = np.array([293.14333333, 140.86333333,  23.46518904,          0. ,          0.,
   0.   ,        0.      ,     0.       ,    0.      ,     0.])
DO_SWRO_minus_std = np.array([ 75.15059429, 121.73493619,  33.18478858 ,          0. ,          0.,
   0.   ,        0.      ,     0.       ,    0.      ,     0.])

CRD_std *= 0.6
DO_std *= 0.6
DO_SWRO_plus_std *= 0.6
DO_SWRO_minus_std *= 0.6
fic_std[0] *= 0.6
PRD_std *= 0.6



# fic_std = genfromtxt('./data/crd_tuning/'+ game + '/FP_std.csv', delimiter=',')
# DO_std = genfromtxt('./data/crd_tuning/'+ game + '/DO_std.csv', delimiter=',')
# CRD_std = genfromtxt('./data/crd_tuning/'+ game + '/CRD_std.csv', delimiter=',')

# axes = plt.gca()
# axes.set_ylim([-0.1,1.25])
#
# fic_mean[0] = 1.9
# DO_mean[0] = 1.9
# CRD_mean[0] = 1.9


X = np.arange(len(CRD_mean)).astype(dtype=np.str)

plt.plot(X, fic_mean, color="C0", marker='o', label='Uniform')
plt.fill_between(X, fic_mean+fic_std, fic_mean-fic_std, alpha=0.1, color="C0")

plt.plot(X, DO_mean, color="C2", marker='o', label='Nash')
plt.fill_between(X, DO_mean+DO_std, DO_mean-DO_std, alpha=0.1, color="C2")

plt.plot(X, PRD_mean, color="C3", marker='o', label='PRD')
plt.fill_between(X, PRD_mean+PRD_std, PRD_mean-PRD_std, alpha=0.1, color="C3")

plt.plot(X, CRD_mean, color="C1", marker='o', label='RRD')
plt.fill_between(X, CRD_mean+CRD_std, CRD_mean-CRD_std, alpha=0.1, color="C1")

# plt.plot(X, DO_SWRO_plus, color="C3", marker='o', label='Nash_SWRO')
# plt.fill_between(X, DO_SWRO_plus+DO_SWRO_plus_std, DO_SWRO_plus-DO_SWRO_plus_std, alpha=0.1, color="C3")
#
# plt.plot(X, DO_SWRO_minus, color="C4", marker='o', label='Nash_MORO')
# plt.fill_between(X, DO_SWRO_minus+DO_SWRO_minus_std, DO_SWRO_minus-DO_SWRO_minus_std, alpha=0.1, color="C4")


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 22)


plt.legend(loc="best", prop={'size': 22})

plt.tight_layout()

plt.show()

