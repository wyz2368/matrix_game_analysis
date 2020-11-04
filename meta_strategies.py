import numpy as np
import collections
from nash_solver.general_nash_solver import gambit_solve
from nash_solver.lp_solver import lp_solve
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator
from utils import *

def double_oracle(meta_games, empirical_games, checkpoint_dir, gambit=False, sample_dev=False):
    """
    Double oracle method.
    :param meta_games:
    :param empirical_games:
    :param checkpoint_dir:
    :param gambit: Whether using gambit as a Nash solver. False： lp solver
    :param sample_dev: Whether sample a beneficial deviation or sample a argmax deviation
    :return:
    """
    num_players = len(meta_games)
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    if gambit:
        # Gambit solver
        nash = gambit_solve(subgames, mode="one", checkpoint_dir=checkpoint_dir[:-1])
    else:
        # LP solver
        nash = lp_solve(subgames)

    # nash = gambit_solve(subgames, mode="one", checkpoint_dir=checkpoint_dir[:-1])

    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    if sample_dev:
        dev_strs, dev_payoff = beneficial_deviation(meta_games, meta_game_nash, nash_payoffs)
        dev_strs, dev_payoff = sample_deviation_strategy(dev_strs, dev_payoff)
    else:
        dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)

    nashconv = 0
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv



def fictitious_play(meta_games, empirical_games, checkpoint_dir=None, sample_dev=False):
    num_strategies, _ = np.shape(meta_games[0])
    subgames = []
    counter0 = collections.Counter(empirical_games[0])
    counter1 = collections.Counter(empirical_games[1])

    idx0 = sorted(list(set(empirical_games[0])))
    idx1 = sorted(list(set(empirical_games[1])))
    idx = np.ix_(idx0, idx1)
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    nash0 = np.ones(len(idx0))
    for i, item in enumerate(idx0):
        nash0[i] = counter0[item]
    nash0 /= np.sum(nash0)

    nash1 = np.ones(len(idx1))
    for i, item in enumerate(idx1):
        nash1[i] = counter1[item]
    nash1 /= np.sum(nash1)
    nash = [nash0, nash1]
    
    nash_payoffs = mixed_strategy_payoff(subgames, nash)

    meta_game_nash = []
    for i, idx in enumerate([idx0, idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, nash[i])
        meta_game_nash.append(ne)

    # dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)
    if sample_dev:
        dev_strs, dev_payoff = beneficial_deviation(meta_games, meta_game_nash, nash_payoffs)
        dev_strs, dev_payoff = sample_deviation_strategy(dev_strs, dev_payoff)
    else:
        dev_strs, dev_payoff = deviation_strategy(meta_games, meta_game_nash)

    nashconv = 0
    for player in range(len(meta_games)):
        nashconv += np.maximum(dev_payoff[player] - nash_payoffs[player], 0)

    return dev_strs, nashconv

def mrcp_solver(meta_games, empirical_games, checkpoint_dir=None, method="alter", recursive=False):
    """
    A wrapper for minimum_regret_profile_calculator, automatically test iterations and clearning remnants mrcp values

    Notice how to deal with closed issue.
    """
    if not hasattr(mrcp_solver, "mrcp_calculator"):
        mrcp_solver.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games, recursive=recursive)
    else:
        # test full game the same
        full_game_different = meta_games[0].shape != mrcp_solver.mrcp_calculator.full_game[0].shape or np.sum(np.absolute(meta_games[0]-mrcp_solver.mrcp_calculator.full_game[0]),axis=None) != 0
        if full_game_different: # change mrcp_calculator
            print('Changing mrcp calculator!!!')
            mrcp_solver.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games, recursive=recursive)
        elif mrcp_solver.mrcp_calculator.mrcp_empirical_game is not None and len(empirical_games[0]) <= len(mrcp_solver.mrcp_calculator.mrcp_empirical_game[0]):
            # another round of random start from full game, might exist potential bugs
            # as I changed _last_empirical_game to mrcp_empirical_game
            print('Clearing the past data from mrcp calculator, should be at the start of a new round')
            mrcp_solver.mrcp_calculator.clear()
        else:
            pass

    # refresh mrcp cache every time calculating MRCP.
    mrcp_solver.mrcp_calculator.clear()
    mrcp_solver.mrcp_calculator(empirical_games)

    num_strategies = meta_games[0].shape[0]
    idx0 = sorted(list(set(mrcp_solver.mrcp_calculator.mrcp_empirical_game[0])))
    idx1 = sorted(list(set(mrcp_solver.mrcp_calculator.mrcp_empirical_game[1])))

    meta_game_nash = []
    for i, idx in enumerate([idx0,idx1]):
        ne = np.zeros(num_strategies)
        np.put(ne, idx, mrcp_solver.mrcp_calculator.mrcp_profile[i])
        meta_game_nash.append(ne)

    # Closeness flag
    closeness = False

    prob1 = meta_game_nash[0]
    prob1 = np.reshape(prob1, newshape=(len(prob1), 1))
    prob2 = meta_game_nash[1]

    payoff_vec0 = np.sum(meta_games[0] * prob2, axis=1)
    payoff_vec0 = np.reshape(payoff_vec0, -1)
    dev_str0 = np.argmax(payoff_vec0)

    payoff_vec1 = np.sum(prob1 * meta_games[1], axis=0)
    payoff_vec1 = np.reshape(payoff_vec1, -1)
    dev_str1 = np.argmax(payoff_vec1)

    if dev_str0 in empirical_games[0] and dev_str1 in empirical_games[1]:
        print("^^^^ Closeness is triggered ^^^^")
        closeness = True

    if closeness:
        payoff_vecs = [payoff_vec0, payoff_vec1]
        dev_strs = closeness_handling(meta_games, empirical_games, payoff_vecs, method, checkpoint_dir)
    else:
        dev_strs = [dev_str0, dev_str1]

    return dev_strs, mrcp_solver.mrcp_calculator.mrcp_value


def closeness_handling(meta_games, empirical_games, payoff_vecs, method, checkpoint_dir):
    """
    Handling the closeness of the MRCP as a MSS.
    :param meta_games:
    :param empirical_games:
    :param method:
    :return:
    """

    if method == "alter":
        return closeness_alter(meta_games, empirical_games, checkpoint_dir)
    elif method == "dev":
        return closeness_dev(empirical_games, payoff_vecs)
    else:
        raise ValueError("Undefined closeness handling method.")


def closeness_dev(empirical_games, payoff_vecs):
    """
    Handling closeness issue by only considering deviation strategies outside
    the empirical game.
    :param meta_games:
    :param empirical_games:
    :param payoff_vecs:
    :return:
    """
    dev_strs = []
    for player, payoff_vec in enumerate(payoff_vecs):
        payoff_vec[list(set(empirical_games[player]))] = -1e5 # mask elements inside empirical game
        dev_strs.append(np.argmax(payoff_vec))

    return dev_strs

def closeness_alter(meta_games, empirical_games, checkpoint_dir):
    """
    Handling closeness issue by only alternating DO and MRCP.
    :param meta_games:
    :param empirical_games:
    :param checkpoint_dir:
    :return:
    """
    dev_strs, _ = double_oracle(meta_games, empirical_games, checkpoint_dir)
    return dev_strs









