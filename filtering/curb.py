"""
Finding the closed under rational behavior (curb).
"""
import numpy as np
from scipy.optimize import linprog

#TODO: have not tested.
def all_conditionally_rational(S, empirical_games_payoff, player):
    """
    Finding all of a player’s best responses conditioned on the belief
    that the other player will play from within some subset of its total strategy space.
    :param S_r:
    :param S_c:
    :param meta_games:
    :param current_player:
    :return:
    """
    S_star = []

    num_strategies_mi = len(S[1-player])
    bounds = [(0, None) for _ in range(num_strategies_mi)]
    A_eq = np.ones(num_strategies_mi)
    b_eq = np.array([1])
    b_ub = np.zeros(num_strategies_mi)
    c = np.zeros(num_strategies_mi)
    for s in S[player]:
        A_ub = []
        if player:
            u_r = empirical_games_payoff[player][:, s]
        else:
            u_r = empirical_games_payoff[player][s, :]
        u_r = np.reshape(u_r, -1)
        for s_p in S[player]:
            if s == s_p:
                continue
            if player:
                u_rp = empirical_games_payoff[player][:, s_p]
            else:
                u_rp = empirical_games_payoff[player][s_p, :]
            u_rp = np.reshape(u_rp, -1)
            A_ub.append(u_rp - u_r)

        A_ub = np.array(A_ub)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        if res["status"] == 0:
            S_star.append(s)

    return S_star

def min_containing_CURB(seed_strategy, empirical_games_payoff):
    """
    The algorithm below finds the smallest CURB set that contains a given seed strategy within a given subgame.
    :return:
    """
    S = [[], [seed_strategy]]
    converge = False
    num_players = len(empirical_games_payoff)
    while not converge:
        converge = True
        for player in range(num_players):
            S_prime = all_conditionally_rational(S, empirical_games_payoff, player=player)
            if len(set(S_prime) - set(S[player])) == 0:
                converge = False
            S[player] = list(set(S[player]).union(set(S_prime)))

    return S











