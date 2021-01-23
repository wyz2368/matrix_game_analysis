import numpy as np
from meta_strategies import prd_solver, iterated_quantal_response_solver
from nash_solver.gambit_tools import load_pkl
from nash_solver.replicator_dynamics_solver import replicator_dynamics

meta_games = load_pkl("./data/meta_game.pkl")
num_strs = np.shape(meta_games[0])[0]

# dev_strs, nashconv = prd_solver(meta_games, [list(range(num_strs)), list(range(num_strs))])
dev_strs, nashconv = iterated_quantal_response_solver(meta_games, [list(range(num_strs)), list(range(num_strs))])

print(nashconv)

