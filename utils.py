import numpy as np
import random
from itertools import product

def set_random_seed(seed=None):
    seed = np.random.randint(low=0,high=1e5) if seed is None else seed
    np.random.seed(seed)
    random.seed(seed)
    return seed


def general_get_joint_strategy_from_marginals(probabilities):
    """Returns a joint strategy matrix from a list of marginals.
    Does not require marginals to have the same lengths.
    Args:
      probabilities: list of probabilities.

    Returns:
      A joint strategy from a list of marginals
    """
    joint = np.outer(probabilities[0], probabilities[1])
    for i in range(len(probabilities) - 2):
        joint = joint.reshape(tuple(list(joint.shape) + [1])) * probabilities[i + 2]
    return joint

def mixed_strategy_payoff(meta_games, probs):
    """
    A multiple player version of mixed strategy payoff writen below by yongzhao
    The lenth of probs could be smaller than that of meta_games
    """
    assert len(meta_games) == len(probs),'number of player not equal'
    for i in range(len(meta_games)):
        assert len(probs[i]) <= meta_games[0].shape[i],'meta game should have larger dimension than marginal probability vector'
    prob_matrix = general_get_joint_strategy_from_marginals(probs)
    prob_slice = tuple([slice(prob_matrix.shape[i]) for i in range(len(meta_games))])
    meta_game_copy = [ele[prob_slice] for ele in meta_games]
    payoffs = []
    for i in range(len(meta_games)):
        payoffs.append(np.sum(meta_game_copy[i]*prob_matrix))
    return payoffs

# This older version of function must be of two players
def mixed_strategy_payoff_2p(meta_games, probs):
   payoffs = []
   prob1 = probs[0]
   prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
   prob2 = probs[1]
   for meta_game in meta_games:
       payoffs.append(np.sum(prob1 * meta_game * prob2))
   return payoffs

def regret_of_variable(prob_var, empirical_games, meta_game):
    """
    Only works for two player case
    Calculate the function value of one data point prob_var
    in amoeba method, Reshape and expand the probability var into full shape
    Input:
        prob_var       : variable that amoeba directly search over
        empirical_games: a list of list, indicating player's strategy sets
        meta_game      : the full game matrix to calculate deviation from
    """
    probs = []
    index = np.cumsum([len(ele) for ele in empirical_games])
    pointer = 0
    for i, idx in enumerate(empirical_games):
        prob = np.zeros(meta_game[0].shape[i])
        np.put(prob, idx, prob_var[pointer:index[i]])
        pointer = index[i]
        probs.append(prob)

    _, dev_payoff = deviation_strategy(meta_game,probs) 
    payoff = mixed_strategy_payoff(meta_game, probs)
    return sum(dev_payoff)-sum(payoff)

def upper_bouned_regret_of_variable(prob_var, empirical_games, meta_game, caches):
    """
        Only works for two player case
        Calculate the upper bounded function value of one data point prob_var
        in amoeba method, Reshape and expand the probability var into full shape.
        Input:
            prob_var       : variable that amoeba directly search over
            empirical_games: a list of list, indicating player's strategy sets
            meta_game      : the full game matrix to calculate deviation from
            caches         : Store the deviation payoff for each player. [cache0, cache1]
    """
    probs = []
    num_player = len(meta_game)
    index = np.cumsum([len(ele) for ele in empirical_games])
    pointer = 0
    for i, idx in enumerate(empirical_games):
        prob = np.zeros(meta_game[0].shape[i])
        np.put(prob, idx, prob_var[pointer:index[i]])
        pointer = index[i]
        probs.append(prob)

    weighted_deviation_payoff = np.zeros(num_player)
    for player in range(num_player):
        for str, i in enumerate(empirical_games[1-player]):
            weighted_deviation_payoff[player] += caches[player].get(str) * probs[1-player][i]

    mixed_payoff = mixed_strategy_payoff_2p(meta_game, probs)

    return sum(weighted_deviation_payoff) - sum(mixed_payoff)

def find_all_deviation_payoffs(empirical_games, meta_game, caches):
    """
    Find all deviation payoff of pure strategy profile. Only need to calculate
    sum_i|S_i| deviations. Only for 2-player game.
    :param empirical_games:
    :param meta_game: the underlying true game
    :param caches: storage of deviation payoffs.
    :return:
    """
    num_strategies_p0 = len(empirical_games[0])
    num_strategies_p1 = len(empirical_games[1])
    if num_strategies_p0 != num_strategies_p1:
        raise ValueError("Currently players have different number of strategies. Haven't supported.")

    # Allow redundant strategies.
    diagonal_profiles = list(zip(empirical_games[0], empirical_games[1]))
    for profile in diagonal_profiles:
        _, payoff = deviation_pure_strategy_profile(meta_game, profile)
        caches[0].save(profile[1], payoff[0])
        caches[1].save(profile[0], payoff[1])

    return caches


class Cache():
    def __init__(self):
        self.cache = {}

    def save(self, key, value):
        if key not in self.cache:
            self.cache[key] = value

    def check(self, key):
        if key not in self.cache:
            return False
        else:
            return self.cache[key]

    def get(self, key):
        return self.cache[key]




def deviation_pure_strategy_profile(meta_games, strategis):
    """
    Find the deviation strategy and payoff for pure strategy profile.
    For 2-player case only.
    :param meta_games: the full game matrix.
    :param strategis: [strategy idx for p1, strategy idx for p2]
    :return:
    """
    dev_strs = []
    dev_strs.append(np.argmax(meta_games[0][:, strategis[1]]))
    dev_strs.append(np.argmax(meta_games[1][strategis[0], :]))

    dev_payoff = [meta_games[0][dev_strs[0], strategis[1]], meta_games[1][strategis[0], dev_strs[1]]]

    return dev_strs, dev_payoff


def deviation_strategy(meta_games, probs):
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    return dev_strs, dev_payoff

def project_onto_unit_simplex(prob):
    """
    Project an n-dim vector prob to the simplex Dn s.t.
    Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}
    :param prob: a numpy array. Each element is a probability.
    :return: projected probability
    """
    prob_length = len(prob)
    bget = False
    sorted_prob = -np.sort(-prob)
    tmpsum = 0

    for i in range(1, prob_length):
        tmpsum = tmpsum + sorted_prob[i-1]
        tmax = (tmpsum - 1) / i
        if tmax >= sorted_prob[i]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + sorted_prob[prob_length-1] - 1) / prob_length

    return np.maximum(0, prob - tmax)


