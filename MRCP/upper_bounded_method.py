"""
This script tests the performance of upper-bounded method of MRCP.
We compare the MRCP regret given by Ameoba method to the regret given by upper-bounded approch.
"""

import numpy as np
from game_generator import Game_generator
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator
from meta_strategies import double_oracle
from nash_solver.gambit_tools import save_pkl

from absl import app
from absl import flags
import datetime
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_strategies", 100, "The number of strategies in full game.")
flags.DEFINE_integer("num_emp_strategies", 40, "The number of strategies in empirical game.")
flags.DEFINE_integer("num_iter", 10, "The number of runs of this test.")
flags.DEFINE_string("game_type", "symmetric_zero_sum", "Type of synthetic game.")


def MRCP_regret_comparison(generator,
                           game_type,
                           empirical_game_size=40,
                           checkpoint_dir=None):
    """
    Compare the MRCP regret given by Ameoba method to the regret given by upper-bounded approach.
    The regret of the NE is listed as a benchmark.
    :param generator: a Game generator
    :param game_type: type of game
    :param empirical_game_size:
    :return:
    """
    if game_type == "zero_sum":
        meta_games = generator.zero_sum_game()
    elif game_type == "general_sum":
        meta_games = generator.general_sum_game()
    elif game_type == "symmetric_zero_sum":
        meta_games = generator.general_sum_game()
    else:
        raise ValueError("Undefined game type.")

    num_total_strategies = len(meta_games[0])
    num_player = len(meta_games)
    empirical_game = []
    for player in range(num_player):
        empirical_game.append(sorted(list(np.random.choice(range(0, num_total_strategies), empirical_game_size))))

    exact_calculator = minimum_regret_profile_calculator(full_game=meta_games)
    appro_calculator = minimum_regret_profile_calculator(full_game=meta_games, approximation=True)
    mrcp_profile, mrcp_value = exact_calculator(empirical_game=empirical_game)
    appro_mrcp_profile, appro_mrcp_value = appro_calculator(empirical_game=empirical_game)

    # Calculate the NE of the empirical game
    _, nashconv = double_oracle(meta_games=meta_games,
                                empirical_games=empirical_game,
                                checkpoint_dir=checkpoint_dir)

    ########## Evaluation ###########
    l2_norm = 0
    for player in range(num_player):
        l2_norm += np.linalg.norm(mrcp_profile[player] - appro_mrcp_profile[player])

    print("The L2 distance is:", l2_norm)
    print("The regret of MRCP:", mrcp_value)
    print("The regret of approximate MRCP:", appro_mrcp_value)
    print("The regret of NE:", nashconv)

    return l2_norm, mrcp_value, appro_mrcp_value, nashconv


def main(argv):
    generator = Game_generator(FLAGS.num_strategies)
    checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir) + '/'

    data = []
    for i in range(FLAGS.num_iter):
        print('################## Iteration {} #################'.format(i))
        l2_norm, mrcp_value, appro_mrcp_value, nashconv = MRCP_regret_comparison(generator=generator,
                                                                                 game_type=FLAGS.game_type,
                                                                                 empirical_game_size=FLAGS.num_emp_strategies,
                                                                                 checkpoint_dir=checkpoint_dir)
        data.append([l2_norm, mrcp_value, appro_mrcp_value, nashconv])

    save_pkl(obj=data, path=checkpoint_dir)


if __name__ == "__main__":
  app.run(main)

