import numpy as np
from meta_strategies import prd_solver, iterated_quantal_response_solver
from nash_solver.gambit_tools import load_pkl
from nash_solver.replicator_dynamics_solver import replicator_dynamics

meta_games = load_pkl("./data/meta_game.pkl")
num_strs = np.shape(meta_games[0])[0]
num_players = 2
print("Num of strs total:", num_strs)

stop = 100
for i in range(1, stop+1):
    print("Current Iteration:", i)
    sub_games = []
    for player in range(num_players):
        sub_games.append(meta_games[player][:i, :i])

    dev_strs, nashconv = prd_solver(sub_games, [list(range(i)), list(range(i))])
    print("Nashconv:", nashconv)

    print("************************************")