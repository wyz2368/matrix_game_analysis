from meta_strategies import double_oracle, fictitious_play, mrcp_solver
from psro_trainer import PSRO_trainer
from utils import deviation_strategy
from utils import mixed_strategy_payoff_2p
from scipy.stats import pearsonr
from nash_solver.gambit_tools import save_pkl

import numpy as np
import functools
print = functools.partial(print, flush=True)


def empirical_game_generator(generator,
                             game_type,
                             meta_method,
                             empirical_game_size,
                             seed=None,
                             checkpoint_dir=None):
    """
    Generate an empirical game which is a subgame of the full matrix game.
    :param generator:
    :param game_type:
    :param seed:
    :param checkpoint_dir:
    :param num_iterations:
    :return:
    """
    # Generate the underlying true game.
    if game_type == "zero_sum":
        meta_games = generator.zero_sum_game()
    elif game_type == "general_sum":
        meta_games = generator.general_sum_game()
    elif game_type == "symmetric_zero_sum":
        meta_games = generator.general_sum_game()
    else:
        raise ValueError("Undefined game type.")

    # A list that records which iteration the empirical game is recorded.
    if empirical_game_size <= 91:
        raise ValueError("The number of sampled EG is large than generated EG.")
    empricial_game_record = list(np.arange(10, 91, 10))

    # Create a meta-trainer.
    if meta_method == "DO":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=generator.num_strategies,
                               num_rounds=1,
                               meta_method=double_oracle,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None)
    elif meta_method == "FP":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=generator.num_strategies,
                               num_rounds=1,
                               meta_method=fictitious_play,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None)
    elif meta_method == "MRCP":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=generator.num_strategies,
                               num_rounds=1,
                               meta_method=mrcp_solver,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None)
    else:
        raise ValueError("Undefined meta-method.")

    trainer.iteration()

    return meta_games, trainer.get_recorded_empirical_game()


def profile_regret(meta_games, strategies):
    """
    Calculate the regret of a strategy profile.
    :param meta_games:
    :param strategies: a list [[], []], extended strategy.
    :return:
    """
    num_players = len(meta_games)
    dev_strs, dev_payoff = deviation_strategy(meta_games, strategies)
    payoffs = mixed_strategy_payoff_2p(meta_games, strategies)

    nashconv = 0
    for player in range(num_players):
        nashconv += np.maximum(dev_payoff[player] - payoffs[player], 0)

    return dev_strs, dev_payoff, nashconv

def extend_prob(probs, empirical_game, meta_games):
    """
    Extend the mixed strategy in the empirical game to the scope of the full game.
    Empirical game is ordered and can be the version after removing redundant strategies or not.
    :param probs:
    :return:
    """
    num_players = len(meta_games)
    strategies = []
    for player in range(num_players):
        strategy = np.zeros(np.shape(meta_games[0])[player])
        for val, i in enumerate(empirical_game[player]):
            strategy[val] += probs[i]
        strategies.append(strategy)

    return strategies

def uniform_simplex_sampling(dim):
    """
    Uniformly sample a unit simplex.
    :param dim: the dimension of sampled vector.
    :return:
    """
    vec = np.random.rand(dim+1)
    vec[0] = 0
    vec[-1] = 1
    vec = np.sort(vec)
    output = np.zeros(dim)
    for i in range(dim):
        output[i] = vec[i+1] - vec[i]

    return output

def sampling_scheme(meta_games, empirical_game, rule, checkpoint_dir=None):
    """
    Sample a strategy profile from the empirical game. Meanwhile, calculate the
    nashconv of the profile and its deviation strategies.
    The MRCP also serves as the evaluation metric.
    :param meta_games:
    :param empirical_game:
    :param rule
    :return:
    """
    if rule == "NE":
        dev_strs, nashconv = double_oracle(meta_games, empirical_game, checkpoint_dir)
    elif rule == "uniform":
        dev_strs, nashconv = fictitious_play(meta_games, empirical_game, checkpoint_dir)
    elif rule == "MRCP":
        dev_strs, nashconv = mrcp_solver(meta_games, empirical_game, checkpoint_dir)
    elif rule == "rand":
        num_players = len(meta_games)
        num_strategies_in_EG = np.shape(empirical_game[0])
        rand_str = []
        for player in range(num_players):
            rand_str.append(uniform_simplex_sampling(num_strategies_in_EG[player]))
        strategies = extend_prob(rand_str, empirical_game, meta_games)
        dev_strs, _, nashconv = profile_regret(meta_games, strategies)
    elif isinstance(rule, list):
        strategies = extend_prob(rule, empirical_game, meta_games)
        dev_strs, _, nashconv = profile_regret(meta_games, strategies)
    else:
        raise ValueError("Undefined Sampling Scheme.")

    return dev_strs, nashconv

def regret_analysis(meta_games, empirical_game, rule, checkpoint_dir=None):
    """
    Analysis on the relationship between regret of profile target and learning performance.
    This function calculates the performance improvement of an empirical game after adding
    a strategy according to the selected profile.
    The performance improvement is the regret decrease of MRCP after adding a strategy.
    :param meta_games:
    :param empirical_game:
    :return:
    """
    num_players = len(meta_games)
    dev_strs, nashconv = sampling_scheme(meta_games, empirical_game, rule, checkpoint_dir)
    _, mrcp_regret_old = sampling_scheme(meta_games, empirical_game, "MRCP", checkpoint_dir)
    for player in range(num_players):
        empirical_game[player].append(dev_strs[player])

    _, mrcp_regret_new = sampling_scheme(meta_games, empirical_game, "MRCP", checkpoint_dir)

    return nashconv, np.maximum(mrcp_regret_old - mrcp_regret_new, 0)

def correlation(a, b):
    """
    Calculate Pearson correlation coefficient and p-value.
    :param a:
    :param b:
    :return:
    """
    corr, p_val = pearsonr(a, b)
    return corr, p_val

def console(generator,
            game_type,
            meta_method,
            empirical_game_size,
            num_samples,
            checkpoint_dir):
    """
    The main entry of the regret analysis.
    :param generator:
    :param game_type:
    :param meta_method:
    :param empirical_game_size:
    :param num_samples:
    :param checkpoint_dir:
    :return:
    """


    meta_games, empirical_games_dict = empirical_game_generator(generator=generator,
                                                                 game_type=game_type,
                                                                 meta_method=meta_method,
                                                                 empirical_game_size=empirical_game_size,
                                                                 seed=None,
                                                                 checkpoint_dir=checkpoint_dir)

    print("Obtain the empirical_games_dict with ", meta_method)
    print("The full game is sample at iteration:", empirical_games_dict.keys())
    print("The number of samples is ", num_samples)

    for key in empirical_games_dict:
        print("############# Iteration {} ############".format(key))
        empirical_games = empirical_games_dict[key]

        regret_of_samples = []
        performance_improvement = []
        for _ in range(num_samples):
            nashconv, improvement = regret_analysis(meta_games,
                                                    empirical_games,
                                                    rule='rand',
                                                    checkpoint_dir=checkpoint_dir)

            regret_of_samples.append(nashconv)
            performance_improvement.append(improvement)

        corr, p_val = correlation(regret_of_samples, performance_improvement)
        print("Correlation coeffient:", corr, "P-value:", p_val)
        save_pkl(obj=regret_of_samples, path=checkpoint_dir + "regret_of_samples_" + str(key))
        save_pkl(obj=performance_improvement, path=checkpoint_dir + "performance_improvement_" + str(key))

        # Compare with standard MSS.
        MSSs = ["NE", "uniform", "MRCP"]
        for mss in MSSs:
            nashconv, improvement = regret_analysis(meta_games,
                                                    empirical_games,
                                                    rule='NE',
                                                    checkpoint_dir=checkpoint_dir)
            print(mss, "--", "regret:", nashconv, "improvement:", improvement)




























