import pickle
from nash_solver.gambit_tools import load_pkl, save_pkl
import numpy as np
from nash_solver.gambit_tools import load_pkl

a1 = load_pkl("./payoff_matrix_att1.pkl")
a2 = load_pkl("./payoff_matrix_att2.pkl")
a3 = load_pkl("./payoff_matrix_att2.pkl")

b1 = load_pkl("./payoff_matrix_def1.pkl")
b2 = load_pkl("./payoff_matrix_def2.pkl")
b3 = load_pkl("./payoff_matrix_def3.pkl")

game = []
defender_payoffs = b1 + b2 + b3
attacker_payoffs = a1 + a2 + a3

game.append(defender_payoffs)
game.append(attacker_payoffs)

game = np.array(game)

# save_pkl(game, '../combined_att_graph.pkl')
