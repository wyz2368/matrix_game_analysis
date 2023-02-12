import numpy as np
from utils import set_random_seed
from meta_strategies import double_oracle
import functools

print = functools.partial(print, flush=True)
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator

""""
MRCP closed temporally with flag #MRCP
"""


class PSRO_trainer(object):
    def __init__(self,
                 meta_games,
                 meta_method,
                 checkpoint_dir,
                 num_iterations=20,
                 empricial_game_record=True,
                 init_strategies=None,
                 seed=None):
        """
        Inputs:
            num_rounds      : repeat psro on matrix games from #num_rounds start points
            meta_method_list: for heuristics block switching
            blocks          : HBS
            empricial_game_record: a list with numbers that indicate the iteration when empirical game is recorded.
            seed            : a integer. If provided, reset every round to guarantee that mrcp calculation is deterministic given an empirical game:
            calculate_neconv   : ne_conv to evaluate to evaluate the heuristics
            calculate_mrcpconv : mrcp_conv to evaluate the heuristics
            init_strategies    : a len(num_rounds) list or a number
        """
        self.meta_games = meta_games
        self.meta_method = meta_method
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        self.init_strategies = init_strategies ##TODO: set to be ZI.

        self.empirical_games = [[], []]
        self.num_iterations = num_iterations

        # Record empirical game.
        self.empricial_game_record = empricial_game_record
        if empricial_game_record is not None:
            self.empirical_games_dict = {}

    def init_round(self, init_strategy):
        self.empirical_games = [[init_strategy], [init_strategy]] #TODO: Check if two players should have different init strategies.

    def iteration(self):

        for it in range(self.num_iterations):
            print('################## Iteration {} ###################'.format(it))
            dev_strs = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)

            self.empirical_games[0].append(dev_strs[0])
            self.empirical_games[0] = sorted(self.empirical_games[0])
            self.empirical_games[1].append(dev_strs[1])
            self.empirical_games[1] = sorted(self.empirical_games[1])

            if self.empricial_game_record is not None and it in self.empricial_game_record:
                self.empirical_games_dict[it] = self.empirical_games.copy()


    def loop(self):
        self.iteration()

    def get_empirical_game(self):
        return self.empirical_games

    def get_recorded_empirical_game(self):
        return self.empirical_games_dict
