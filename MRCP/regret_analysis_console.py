from game_generator import Game_generator
from MRCP.regret_analysis import console

from absl import app
from absl import flags
import os
import sys
import datetime
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_strategies", 200, "The number of strategies in full game.")
flags.DEFINE_integer("num_emp_strategies", 100, "The number of strategies in empirical game.")
flags.DEFINE_integer("num_samples", 2000, "The number of sampled strategy profiles.")
flags.DEFINE_string("game_type", "symmetric_zero_sum", "Type of synthetic game.")
flags.DEFINE_string("meta_method", "DO", "Meta method for game generation")


def main(argv):
    generator = Game_generator(FLAGS.num_strategies)

    root_path = './' + FLAGS.game_type + "_" + FLAGS.meta_method + '/'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), root_path, checkpoint_dir) + '_regret/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

    console(generator=generator,
            game_type=FLAGS.game_type,
            meta_method=FLAGS.meta_method,
            empirical_game_size=FLAGS.num_emp_strategies,
            num_samples=FLAGS.num_samples,
            checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
  app.run(main)
