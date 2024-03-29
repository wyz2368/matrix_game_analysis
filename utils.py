import numpy as np
import random
import functools
from scipy.stats import entropy
print = functools.partial(print, flush=True)

def set_random_seed(seed=None):
    seed = np.random.randint(low=0,high=1e5) if seed is None else seed
    np.random.seed(seed)
    random.seed(seed)
    return seed

def profile_entropy(profile):
    ent = 0
    for prob in profile:
        ent += entropy(prob)
    return ent


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
        payoffs.append(np.sum(meta_game_copy[i] * prob_matrix))
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

def regret_of_variable(prob_var, empirical_games, meta_game, sum_regret=True):
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

    _, dev_payoff = deviation_strategy(meta_game, probs)
    payoff = mixed_strategy_payoff(meta_game, probs)
    if sum_regret:
        return sum(dev_payoff)-sum(payoff)
    else:
        return np.max(dev_payoff - np.array(payoff))

def upper_bouned_regret_of_variable(prob_var, empirical_games, meta_game, caches, discount=0.05):
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
        for i, str in enumerate(empirical_games[1-player]):
            weighted_deviation_payoff[player] += caches[player].get(str) * prob_var[i + index[0] * (1-player)]

    mixed_payoff = mixed_strategy_payoff_2p(meta_game, probs)

    # return np.sum(np.log(np.maximum(weighted_deviation_payoff - np.array(mixed_payoff) - discount * profile_entropy(probs), 0)))
    return np.max(np.maximum(weighted_deviation_payoff - np.array(mixed_payoff) - discount * profile_entropy(probs), 0))

    # return np.sum(weighted_deviation_payoff - np.array(mixed_payoff))

def sampled_bouned_regret_of_variable(prob_var, empirical_games, meta_game, caches=None, discount=0.05):
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

    deviation_payoff_in_EG = deviation_within_EG(meta_game, empirical_games, probs)

    weighted_deviation_payoff = np.zeros(num_player, dtype=np.float32)
    for player in range(num_player):
        for i, str in enumerate(empirical_games[1-player]):
            payoff_vec = benefitial_deviation_pure_strategy_profile(meta_game, opponent=1-player, strategy=str, base_value=deviation_payoff_in_EG)
            if len(payoff_vec) == 0:
                weighted_deviation_payoff[player] += deviation_payoff_in_EG[player] * prob_var[i + index[0] * (1 - player)]
            else:
                weighted_deviation_payoff[player] += np.min(payoff_vec) * prob_var[i + index[0] * (1 - player)]

    mixed_payoff = mixed_strategy_payoff_2p(meta_game, probs)

    # print("weighted_deviation_payoff:", weighted_deviation_payoff)
    # print("mixed_payoff:", mixed_payoff)

    return np.max(np.maximum(weighted_deviation_payoff - np.array(mixed_payoff) - discount * profile_entropy(probs), 0))

    # return np.sum(weighted_deviation_payoff - np.array(mixed_payoff))

def find_all_deviation_payoffs(empirical_games, meta_game, caches):
    """
    Find all deviation payoff of pure strategy profile. Only need to calculate
    sum_i|S_i| deviations. Only for 2-player game.
    :param empirical_games:
    :param meta_game: the underlying true game
    :param caches: storage of deviation payoffs.
    :param mean:
    :return:
    """
    num_strategies_p0 = len(empirical_games[0])
    num_strategies_p1 = len(empirical_games[1])

    if num_strategies_p0 != num_strategies_p1:
        raise ValueError("Haven't supported that 2 players have different number of strategies.")

    # Allow redundant strategies.
    diagonal_profiles = list(zip(empirical_games[0], empirical_games[1]))
    for profile in diagonal_profiles:
        _, payoff = deviation_pure_strategy_profile(meta_game, profile)
        caches[0].save(key=profile[1], value=payoff[0])
        caches[1].save(key=profile[0], value=payoff[1])

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

def benefitial_deviation_pure_strategy_profile(meta_games, opponent, strategy, base_value):
    """
    Find the deviation strategy and payoff for pure strategy profile.
    For 2-player case only.
    :param meta_games: the full game matrix.
    :param strategis: [strategy idx for p1, strategy idx for p2]
    :return:
    """
    if opponent == 1:
        payoff_vec = meta_games[0][:, strategy]
        payoff_vec = np.reshape(payoff_vec, -1)
        idx = list(np.where(payoff_vec > base_value[0])[0])

    else:
        payoff_vec = meta_games[1][strategy, :]
        payoff_vec = np.reshape(payoff_vec, -1)
        idx = list(np.where(payoff_vec > base_value[1])[0])

    return payoff_vec[idx]


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

def deviation_strategy_with_objective(meta_games, probs, alpha=1.0, minus=False):
    """
    This is for different response oracle.
    :param meta_games:
    :param probs:
    :return:
    """
    print("Deviation_strategy_with_objective is working in the model minus = ", minus, " with alpha = ", alpha)
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec_self = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec_other_player = np.sum(meta_games[1] * prob2, axis=1)
    if minus:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) - (1 - alpha) * np.reshape(payoff_vec_other_player, -1)
    else:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) + (1 - alpha) * np.reshape(payoff_vec_other_player, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec_self[idx])

    payoff_vec_self = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec_other_player = np.sum(prob1 * meta_games[0], axis=0)
    if minus:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) - (1 - alpha) * np.reshape(payoff_vec_other_player, -1)
    else:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) + (1 - alpha) * np.reshape(payoff_vec_other_player, -1)

    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec_self[idx])

    return dev_strs, dev_payoff


def deviation_strategy_with_NBS(meta_games, probs, alpha=1.0, minus=False):
    """
    This is for different response oracle.
    :param meta_games:
    :param probs:
    :return:
    """
    print("Deviation_strategy_with_NBS is working in the model minus = ", minus, " with alpha = ", alpha)
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec_self = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec_other_player = np.sum(meta_games[1] * prob2, axis=1)
    if minus:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) - (1 - alpha) * np.reshape(payoff_vec_other_player, -1)
    else:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) + (1 - alpha) * np.reshape(payoff_vec_other_player, -1) * np.reshape(payoff_vec_self, -1)
    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec_self[idx])

    payoff_vec_self = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec_other_player = np.sum(prob1 * meta_games[0], axis=0)
    if minus:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) - (1 - alpha) * np.reshape(payoff_vec_other_player, -1)
    else:
        payoff_vec = alpha * np.reshape(payoff_vec_self, -1) + (1 - alpha) * np.reshape(payoff_vec_other_player, -1)

    idx = np.argmax(payoff_vec)
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec_self[idx])

    return dev_strs, dev_payoff

# Functions for 3-player games.
def dev_regret_general(meta_games, probs):
    """
        Calculate the regret of a profile in an empirical game with any number of players.
        :param meta_games:
        :param probs: a strategy profile
        :return:
        """
    num_players = len(meta_games)
    num_strategies = np.shape(meta_games[0])

    prob_matrix = general_get_joint_strategy_from_marginals(probs)
    deviation_payoffs = []

    for i in range(num_players):
        profile_payoff = np.sum(meta_games[i] * prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(num_strategies[i]):
            dev_prob = probs.copy()
            dev_prob[i] = np.zeros(num_strategies[i])
            dev_prob[i][j] = 1
            new_prob_matrix = general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i] * new_prob_matrix))
        deviation_payoffs.append(dev_payoff - profile_payoff)

    dev_strs = [np.argmax(ele) for ele in deviation_payoffs]
    dev_payoff = []
    for idx, str in enumerate(dev_strs):
        dev_payoff.append(deviation_payoffs[idx][str])

    nashconv = np.sum([np.max(ele) for ele in deviation_payoffs])

    return dev_strs, dev_payoff, nashconv

    # nashconv = np.sum([np.max(ele) for ele in deviation_payoffs])
    # return nashconv


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

def beneficial_deviation(meta_games, probs, base_value):
    """
    Find all beneficial deviations and corresponding payoffs.
    Only for two-player case.
    :param meta_games:
    :param probs:
    :param base_value: deviation beyond this value. [p1, p2]
    :return:
    """
    dev_strs = []
    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = list(np.where(payoff_vec > base_value[0])[0])
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    idx = list(np.where(payoff_vec > base_value[1])[0])
    dev_strs.append(idx)
    dev_payoff.append(payoff_vec[idx])

    return dev_strs, dev_payoff

def sample_deviation_strategy(dev_strs, dev_payoff):
    """
    Sample a deviation strategy and corresponding payoff.
    :param dev_strs:
    :param dev_payoff:
    :return:
    """
    num_players = len(dev_strs)
    num_deviations = len(dev_strs[0])
    sampled_str = []
    sample_payoff = []

    for player in range(num_players):
        idx = np.random.choice(np.arange(num_deviations))
        sampled_str.append(dev_strs[player][idx])
        sample_payoff.append(dev_payoff[player][idx])

    return sampled_str, sample_payoff

def deviation_within_EG(meta_games, empirical_games, probs):
    """
    Calculate the deviation payoff of a profile within a empirical game.
    :param meta_games:
    :param empirical_games:
    :param probs:
    :return:
    """

    dev_payoff = []
    prob1 = probs[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = probs[1]

    payoff_vec = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec = np.reshape(payoff_vec, -1)
    payoff_vec_EG = payoff_vec[empirical_games[0]]
    dev_payoff.append(np.max(payoff_vec_EG))

    payoff_vec = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec = np.reshape(payoff_vec, -1)
    payoff_vec_EG = payoff_vec[empirical_games[1]]
    dev_payoff.append(np.max(payoff_vec_EG))

    return dev_payoff
