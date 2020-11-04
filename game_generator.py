import numpy as np
from nash_solver.gambit_tools import load_pkl

class Game_generator(object):
    def __init__(self,
                 num_strategies,
                 num_players=2,
                 payoff_ub=501,
                 payoff_lb=-500):
        self.num_strategies = num_strategies
        self.num_players = num_players
        assert num_players == 2
        self.payoff_ub = payoff_ub
        self.payoff_lb = payoff_lb

    def zero_sum_game(self):
        meta_game = np.random.randint(low=self.payoff_lb,
                                      high=self.payoff_ub,
                                      size=(self.num_strategies, self.num_strategies))
        return [meta_game, -meta_game]

    def general_sum_game(self):
        meta_game1 = np.random.randint(low=self.payoff_lb,
                                      high=self.payoff_ub,
                                      size=(self.num_strategies, self.num_strategies))
        meta_game2 = np.random.randint(low=self.payoff_lb,
                                       high=self.payoff_ub,
                                       size=(self.num_strategies, self.num_strategies))
        return [meta_game1, meta_game2]

    def symmetric_zero_sum_game(self):
        meta_game = np.random.randint(low=self.payoff_lb/2,
                                      high=np.ceil(self.payoff_ub/2),
                                      size=(self.num_strategies, self.num_strategies))
        meta_game += meta_game.T
        return [meta_game, -meta_game]

    def real_world_game(self, game_name):
        meta_games = load_pkl("./real_world_games/real_world_meta_games.pkl")
        return meta_games[game_name]

    def transitive_game(self):
        raise NotImplementedError

    def cyclic_game(self):
        raise NotImplementedError
