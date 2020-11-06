from nash_solver.lp_solver import lp_solve
import numpy as np
import copy

def nash_clustering(meta_games, threshold=1e-3):
    """
    Find nash clustering according to Spinning Top paper.
    Notice that currently we don't use maxent NE.
    For 2-player zero-sum game.
    :param meta_games: game payoff for clustering
    :return:
    """
    curr_games = copy.copy(meta_games)
    nash_clusters = []

    num_strategies = list(np.shape(meta_games[0]))
    original_idx = [list(range(num)) for num in num_strategies]

    while(True):
        nash = lp_solve(curr_games)
        idx = [list(np.where(ne > threshold)[0]) for ne in nash]
        nash_clusters.append([])
        for player, orig_idx in enumerate(original_idx):
            nash_clusters[-1].append([orig_idx[i] for i in idx[player]])
            orig_idx = list(np.delete(orig_idx, idx[player]))
            original_idx[player] = orig_idx

        curr_games = remove_clusters(curr_games, idx)
        if curr_games is None:
            break

    return nash_clusters


def remove_clusters(meta_games, idx):
    """
    Remove the corresponding index from a matrix.
    :param meta_games:
    :param idx: a list [[1,2,3], [1,2,3]]
    :return:
    """

    num_strategies = list(np.shape(meta_games[0]))
    new_idx = []
    for player, num in enumerate(num_strategies):
        new_idx.append(sorted(list(set(range(num)) - set(idx[player]))))

    if len(new_idx[0]) == 0 or len(new_idx[1]) == 0:
        return None

    idx = np.ix_(new_idx[0], new_idx[1])
    subgames = []
    for meta_game in meta_games:
        subgames.append(meta_game[idx])

    return subgames