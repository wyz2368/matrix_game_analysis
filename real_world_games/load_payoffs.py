import pickle
from nash_solver.gambit_tools import load_pkl, save_pkl
import numpy as np
import copy

# Load payoffs
with open("./spinning_top_payoffs.pkl", "rb") as fh:
  payoffs = pickle.load(fh)

real_world_meta_games = copy.copy(payoffs)
# Iterate over games
print("======================================================")
for game_name in payoffs:
    real_world_meta_games[game_name] = [payoffs[game_name], -payoffs[game_name]]

    print(f"Game name: {game_name}")
    print(f"Number of strategies: {payoffs[game_name].shape[0]}")
    print(f"Shape of the payoff matrix: {payoffs[game_name].shape}")
    print("======================================================")


  # # Sort strategies by mean winrate for nice presentation
  # order = np.argsort(-payoffs[game_name].mean(1))
  #
  # # Plot the payoff
  # plt.figure()
  # plt.title(game_name)
  # plt.imshow(payoffs[game_name][order, :][:, order])
  # plt.axis('off')
  # plt.show()
  # plt.close()

# print([payoffs["RPS"], -payoffs["RPS"]])
# save_pkl(obj=real_world_meta_games, path="./real_world_meta_games.pkl")
# meta_games = load_pkl("./real_world_meta_games.pkl")
# game = meta_games['Kuhn-poker']
# print(np.shape(game))
# print(game[0])
# print(list(meta_games.keys()))










# ======================================================
# Game name: 10,3-Blotto
# Number of strategies: 66
# Shape of the payoff matrix: (66, 66)
# ======================================================
# Game name: 10,4-Blotto
# Number of strategies: 286
# Shape of the payoff matrix: (286, 286)
# ======================================================
# Game name: 10,5-Blotto
# Number of strategies: 1001
# Shape of the payoff matrix: (1001, 1001)
# ======================================================
# Game name: 3-move parity game 2
# Number of strategies: 160
# Shape of the payoff matrix: (160, 160)
# ======================================================
# Game name: 5,3-Blotto
# Number of strategies: 21
# Shape of the payoff matrix: (21, 21)
# ======================================================
# Game name: 5,4-Blotto
# Number of strategies: 56
# Shape of the payoff matrix: (56, 56)
# ======================================================
# Game name: 5,5-Blotto
# Number of strategies: 126
# Shape of the payoff matrix: (126, 126)
# ======================================================
# Game name: AlphaStar
# Number of strategies: 888
# Shape of the payoff matrix: (888, 888)
# ======================================================
# Game name: Blotto
# Number of strategies: 1001
# Shape of the payoff matrix: (1001, 1001)
# ======================================================
# Game name: Disc game
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Elo game + noise=0.1
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Elo game + noise=0.5
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Elo game + noise=1.0
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Elo game
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Kuhn-poker
# Number of strategies: 64
# Shape of the payoff matrix: (64, 64)
# ======================================================
# Game name: Normal Bernoulli game
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: RPS
# Number of strategies: 3
# Shape of the payoff matrix: (3, 3)
# ======================================================
# Game name: Random game of skill
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Transitive game
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: Triangular game
# Number of strategies: 1000
# Shape of the payoff matrix: (1000, 1000)
# ======================================================
# Game name: connect_four
# Number of strategies: 1470
# Shape of the payoff matrix: (1470, 1470)
# ======================================================
# Game name: go(board_size=3,komi=6.5)
# Number of strategies: 1933
# Shape of the payoff matrix: (1933, 1933)
# ======================================================
# Game name: go(board_size=4,komi=6.5)
# Number of strategies: 1679
# Shape of the payoff matrix: (1679, 1679)
# ======================================================
# Game name: hex(board_size=3)
# Number of strategies: 766
# Shape of the payoff matrix: (766, 766)
# ======================================================
# Game name: misere(game=tic_tac_toe())
# Number of strategies: 926
# Shape of the payoff matrix: (926, 926)
# ======================================================
# Game name: quoridor(board_size=3)
# Number of strategies: 1404
# Shape of the payoff matrix: (1404, 1404)
# ======================================================
# Game name: quoridor(board_size=4)
# Number of strategies: 1540
# Shape of the payoff matrix: (1540, 1540)
# ======================================================
# Game name: tic_tac_toe
# Number of strategies: 880
# Shape of the payoff matrix: (880, 880)
# ======================================================