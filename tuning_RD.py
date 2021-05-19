import numpy as np
from meta_strategies import prd_solver, iterated_quantal_response_solver
from nash_solver.gambit_tools import load_pkl
from nash_solver.replicator_dynamics_solver import replicator_dynamics

meta_games = load_pkl("./data/meta_game.pkl")
num_strs = np.shape(meta_games[0])[0]
print("Num of strs:", num_strs)

sub_games = []
i = 30
for player in range(3):
    sub_games.append(meta_games[player][:i, :i, :i])

dev_strs, nashconv = prd_solver(sub_games, [list(range(i)), list(range(i)), list(range(i))])
# dev_strs, nashconv = iterated_quantal_response_solver(meta_games, [list(range(num_strs)), list(range(num_strs))])

print(nashconv)

