import pickle
from nash_solver.gambit_tools import load_pkl, save_pkl
import numpy as np
import copy

# # Load payoffs
# with open("./spinning_top_payoffs.pkl", "rb") as fh:
#   payoffs = pickle.load(fh)
#
# # Iterate over games
# print("======================================================")
# for game_name in payoffs:
#     # real_world_meta_games[game_name] = [payoffs[game_name], -payoffs[game_name]]
#
#   print(f"Game name: {game_name}")
#   print(f"Number of strategies: {payoffs[game_name].shape[0]}")
#   print(f"Shape of the payoff matrix: {payoffs[game_name].shape}")
#   print("======================================================")
#   print()

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
meta_games = load_pkl("./real_world_meta_games.pkl")
print(meta_games["AlphaStar"][0][-1])