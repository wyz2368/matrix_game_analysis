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

# This file plots the combined game for Leduc poker in the evaluation strategy exploration paper.

plt.figure()
# plt.title("NashConv Curves ", fontsize = 22)


DO_br_mean = genfromtxt('../data/combined_game_data/DO_br_mean.csv', delimiter=',')
PRD_br_mean = genfromtxt('../data/combined_game_data/PRD_br_mean.csv', delimiter=',')
FIC_br_mean = genfromtxt('../data/combined_game_data/FIC_br_mean.csv', delimiter=',')

DO_cg_mean = genfromtxt('../data/combined_game_data/DO_cg_mean.csv', delimiter=',')
PRD_cg_mean = genfromtxt('../data/combined_game_data/PRD_cg_mean.csv', delimiter=',')
FIC_cg_mean = genfromtxt('../data/combined_game_data/FIC_cg_mean.csv', delimiter=',')

DO_br_std = genfromtxt('../data/combined_game_data/DO_br_std.csv', delimiter=',')
PRD_br_std = genfromtxt('../data/combined_game_data/PRD_br_std.csv', delimiter=',')
FIC_br_std = genfromtxt('../data/combined_game_data/FIC_br_std.csv', delimiter=',')

DO_cg_std = genfromtxt('../data/combined_game_data/DO_cg_std.csv', delimiter=',')
PRD_cg_std = genfromtxt('../data/combined_game_data/PRD_cg_std.csv', delimiter=',')
FIC_cg_std = genfromtxt('../data/combined_game_data/FIC_cg_std.csv', delimiter=',')

axes = plt.gca()
axes.set_ylim([0,3])

X = np.arange(1, 150)

plt.plot(X, DO_br_mean, color="blue", label='DO w. BR')
plt.fill_between(X, DO_br_mean+DO_br_std, DO_br_mean-DO_br_std, alpha=0.1, color="blue")

plt.plot(X, PRD_br_mean, color="C2", label='PRD w. BR')
plt.fill_between(X, PRD_br_mean+PRD_br_std, PRD_br_mean-PRD_br_std, alpha=0.1, color="C2")

plt.plot(X, FIC_br_mean, color="C3", label='FP w. BR')
plt.fill_between(X, FIC_br_mean+FIC_br_std, FIC_br_mean-FIC_br_std, alpha=0.1, color="C3")

plt.plot(X, DO_cg_mean, color="C4", label='DO w. combined game')
plt.fill_between(X, DO_cg_mean+DO_cg_std, DO_cg_mean-DO_cg_std, alpha=0.1, color="C4")
#
plt.plot(X, PRD_cg_mean, color="C9", label='PRD w. combined game')
plt.fill_between(X, PRD_cg_mean+PRD_cg_std, PRD_cg_mean-PRD_cg_std, alpha=0.1, color="C9")
#
plt.plot(X, FIC_cg_mean, color="C1", label='FP w. combined game')
plt.fill_between(X, FIC_cg_mean+FIC_cg_std, FIC_cg_mean-FIC_cg_std, alpha=0.1, color="C1")




plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('Regret', fontsize = 19)

plt.legend(loc="best", prop={'size': 16})

plt.show()

