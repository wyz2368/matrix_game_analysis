from game_generator import Game_generator
from MRCP.regret_analysis_real import console
from nash_solver.gambit_tools import load_pkl

from absl import app
from absl import flags
import numpy as np
import os
import sys
import datetime
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_emp_strategies", 110, "The number of strategies in empirical game.")
flags.DEFINE_integer("num_samples", 200, "The number of sampled strategy profiles.")
flags.DEFINE_string("game_type", "symmetric_zero_sum", "Type of synthetic game.")
flags.DEFINE_string("meta_method", "DO", "Meta method for game generation")


def main(argv):
    seed = np.random.randint(low=0, high=1e5)

    root_path = './' + FLAGS.game_type + "_" + FLAGS.meta_method + '_regret/'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    real_world_meta_games = load_pkl('../real_world_games/real_world_meta_games.pkl')

    game_types = ['10,4-Blotto', 'AlphaStar', 'Kuhn-poker', 'Random game of skill', 'Transitive game',
                  'connect_four', 'quoridor(board_size=4)', ' misere(game=tic_tac_toe())', 'hex(board_size=3)',
                  'go(board_size=4,komi=6.5)']

    for game_type in game_types:
        checkpoint_dir = os.path.join(os.getcwd(), root_path) + game_type + '_' + str(seed)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

        print("================================================")
        print("======The current game is ", game_type, "=========")
        print("================================================")

        console(meta_games=real_world_meta_games[game_type],
                meta_method=FLAGS.meta_method,
                empirical_game_size=FLAGS.num_emp_strategies,
                num_samples=FLAGS.num_samples,
                checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
  app.run(main)
