import numpy as np


class Bargaining():
    def __init__(self, offer_dim, offer_max, time_horizon):
        self._offer_dim = offer_dim
        self._offer_max = offer_max

        if len(self._offer_max) != self._offer_dim:
            raise ValueError

        self._time_horizon = time_horizon

        self._current_offer = np.zeros(self._offer_dim)
        self._current_time = 0

        self._opponent_strategy = None

    def set_opponent_strategy(self, strategy):
        self._opponent_strategy = strategy

    def set_utility_function(self, utility_function):
        self._utility_function = utility_function

    def step(self, action):
        if self._opponent_strategy is None:
            raise ValueError

        if self._current_time == self._time_horizon or action[0] == 1:
            return self._current_offer, True

        self._current_offer = action[1:]

        return self._current_offer, False

    def reset(self):
        self._current_offer = np.zeros(self._offer_dim)
        self._current_time = 0
        self._opponent_strategy = None




