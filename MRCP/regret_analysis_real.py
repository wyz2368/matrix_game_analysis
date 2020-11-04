from meta_strategies import double_oracle, fictitious_play, mrcp_solver
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator
from psro_trainer import PSRO_trainer
from utils import deviation_strategy
from utils import mixed_strategy_payoff_2p
from scipy.stats import pearsonr
from nash_solver.gambit_tools import save_pkl, load_pkl

import numpy as np
import copy
import functools
print = functools.partial(print, flush=True)


def empirical_game_generator(meta_games,
                             meta_method,
                             empirical_game_size,
                             seed=None,
                             checkpoint_dir=None):
    """
    Generate an empirical game which is a subgame of the full matrix game.
    :param generator:  a Game generator
    :param game_type: type of the game, options: "symmetric_zero_sum", "zero_sum", "general_sum"
    :param meta_method: Method for generating the empirical game.
    :param empirical_game_size:
    :param seed: random seed
    :param checkpoint_dir:
    :return:
    """

    # Assume players have the same number of strategies.
    num_strategies = np.shape(meta_games[0])[0]

    # A list that records which iteration the empirical game is recorded.
    if empirical_game_size > num_strategies:
        raise ValueError("The size of EG is large than the full game.")

    empricial_game_record = list(range(10, 101, 10))

    if empirical_game_size < max(empricial_game_record):
        raise ValueError("The number of sampled EG is large than generated EG.")

    # Create a meta-trainer.
    if meta_method == "DO":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=num_strategies,
                               num_rounds=1,
                               meta_method=double_oracle,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None,
                               calculate_neconv=False,
                               calculate_mrcpconv=False
                               )
    elif meta_method == "FP":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=num_strategies,
                               num_rounds=1,
                               meta_method=fictitious_play,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None,
                               calculate_neconv=False,
                               calculate_mrcpconv=False
                               )
    elif meta_method == "MRCP":
        trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=num_strategies,
                               num_rounds=1,
                               meta_method=mrcp_solver,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=empirical_game_size,
                               empricial_game_record=empricial_game_record,
                               seed=seed,
                               init_strategies=None,
                               calculate_neconv=False,
                               calculate_mrcpconv=False
                               )
    else:
        raise ValueError("Undefined meta-method.")

    # Don't use trainer.iteration() since the empirical game won't be initialized.
    trainer.loop()

    return trainer.get_recorded_empirical_game()


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
        for i, val in enumerate(empirical_game[player]):
            strategy[val] += probs[player][i]
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
    sampled_profile = None
    if rule == "NE":
        dev_strs, nashconv = double_oracle(meta_games, empirical_game, checkpoint_dir)
    elif rule == "uniform":
        dev_strs, nashconv = fictitious_play(meta_games, empirical_game, checkpoint_dir)
    elif rule == "MRCP":
        dev_strs, nashconv = mrcp_solver(meta_games, empirical_game, checkpoint_dir)
    elif rule == "rand":
        num_players = len(meta_games)
        rand_str = []
        for player in range(num_players):
            num_strategies_in_EG = len(empirical_game[player])
            rand_str.append(uniform_simplex_sampling(num_strategies_in_EG))
        sampled_profile = rand_str
        strategies = extend_prob(rand_str, empirical_game, meta_games)
        dev_strs, _, nashconv = profile_regret(meta_games, strategies)
    elif isinstance(rule, list):
        # For inputting a strategy.
        strategies = extend_prob(rule, empirical_game, meta_games)
        dev_strs, _, nashconv = profile_regret(meta_games, strategies)
    else:
        raise ValueError("Undefined Sampling Scheme.")

    return dev_strs, nashconv, sampled_profile

def regret_analysis(meta_games,
                    empirical_game,
                    rule,
                    MRCP_calculator,
                    mrcp_regret_old,
                    checkpoint_dir=None):
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
    dev_strs, nashconv, sampled_profile = sampling_scheme(meta_games, empirical_game, rule, checkpoint_dir)

    # _, mrcp_regret_old = MRCP_calculator(empirical_game=empirical_game)
    # _, mrcp_regret_old = sampling_scheme(meta_games, empirical_game, "MRCP", checkpoint_dir)

    # Add a mechanism to detect repeated strategies.
    if dev_strs[0] in empirical_game[0] and dev_strs[1] in empirical_game[1]:
        print("No empirical game change.")
        return nashconv, 0, None

    print("The old empirical game is ", empirical_game)
    copied_empirical_game = copy.deepcopy(empirical_game)
    for player in range(num_players):
        copied_empirical_game[player].append(dev_strs[player])
    print("The new empirical game is ", copied_empirical_game)

    MRCP_calculator.clear()
    _, mrcp_regret_new = MRCP_calculator(empirical_game=copied_empirical_game)

    # _, mrcp_regret_new = sampling_scheme(meta_games, empirical_game, "MRCP", checkpoint_dir)
    print("The nashconv of the sample is ", nashconv)
    print("mrcp_regret_old - mrcp_regret_new=", mrcp_regret_old - mrcp_regret_new, " : ", mrcp_regret_old, mrcp_regret_new)

    return nashconv, np.maximum(mrcp_regret_old - mrcp_regret_new, 0), sampled_profile

def correlation(a, b):
    """
    Calculate Pearson correlation coefficient and p-value.
    :param a:
    :param b:
    :return:
    """
    corr, p_val = pearsonr(a, b)
    return corr, p_val

def console(meta_games,
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


    empirical_games_dict = empirical_game_generator(meta_games=meta_games,
                                                     meta_method=meta_method,
                                                     empirical_game_size=empirical_game_size,
                                                     seed=None,
                                                     checkpoint_dir=checkpoint_dir)

    print("Obtain the empirical_games_dict with ", meta_method)
    print("The full game is sample at iteration:", empirical_games_dict.keys())
    print("The number of samples is ", num_samples)

    exact_calculator = minimum_regret_profile_calculator(full_game=meta_games)

    for key in empirical_games_dict:
        print("############# Iteration {} ############".format(key))
        empirical_games = empirical_games_dict[key]
        copied_EG = copy.deepcopy(empirical_games)


        # Remove repeated strategies.
        idx0 = sorted(list(set(empirical_games[0])))
        idx1 = sorted(list(set(empirical_games[1])))
        empirical_games = [idx0, idx1]

        # Calculate the MRCP regret of the empirical game.
        _, mrcp_regret_old = exact_calculator(empirical_game=empirical_games)

        # Compare with standard MSS.
        MSS_regret_of_samples = []
        MSS_performance_improvement = []
        MSSs = ["NE", "MRCP"]
        for mss in MSSs:
            nashconv, improvement, _ = regret_analysis(meta_games,
                                                    empirical_games,
                                                    rule=mss,
                                                    MRCP_calculator=exact_calculator,
                                                    mrcp_regret_old=mrcp_regret_old,
                                                    checkpoint_dir=checkpoint_dir)
            print(mss, "--", "regret:", nashconv, "improvement:", improvement)
            MSS_regret_of_samples.append(nashconv)
            MSS_performance_improvement.append(improvement)

        # Special for fictitious play.
        nashconv, improvement, _ = regret_analysis(meta_games,
                                                copied_EG,
                                                rule="uniform",
                                                MRCP_calculator=exact_calculator,
                                                mrcp_regret_old=mrcp_regret_old,
                                                checkpoint_dir=checkpoint_dir)
        print("uniform", "--", "regret:", nashconv, "improvement:", improvement)
        MSS_regret_of_samples.append(nashconv)
        MSS_performance_improvement.append(improvement)

        print("----------- -------------- ----------")
        print("----------- Start Sampling ----------")
        print("----------- -------------- ----------")

        regret_of_samples = []
        performance_improvement = []
        sampled_profiles = []
        better_than_NE_idx = []
        cnt_zeros = 0
        for i in range(num_samples):
            print("------Sample #", i, "------")
            nashconv, improvement, sampled_profile = regret_analysis(meta_games,
                                                                    empirical_games,
                                                                    rule='rand',
                                                                    MRCP_calculator=exact_calculator,
                                                                    mrcp_regret_old=mrcp_regret_old,
                                                                    checkpoint_dir=checkpoint_dir)

            if np.abs(improvement) < 1e-5:
                cnt_zeros += 1
            regret_of_samples.append(nashconv)
            performance_improvement.append(improvement)
            sampled_profiles.append(sampled_profile)

            if improvement > MSS_performance_improvement[0]:
                better_than_NE_idx.append(i)


        corr, p_val = correlation(regret_of_samples, performance_improvement)
        print("Correlation coeffient:", corr, "P-value:", p_val)
        print("Number of zero improvement is ", cnt_zeros)
        save_pkl(obj=regret_of_samples, path=checkpoint_dir + "regret_of_samples_" + str(key) + ".pkl")
        save_pkl(obj=performance_improvement, path=checkpoint_dir + "performance_improvement_" + str(key) + ".pkl")
        save_pkl(obj=empirical_games, path=checkpoint_dir + "empirical_games_" + str(key) + ".pkl")
        save_pkl(obj=MSS_regret_of_samples, path=checkpoint_dir + "MSS_regret_of_samples_" + str(key) + ".pkl")
        save_pkl(obj=MSS_performance_improvement, path=checkpoint_dir + "MSS_performance_improvement_" + str(key) + ".pkl")

        save_pkl(obj=sampled_profiles, path=checkpoint_dir + "sampled_profiles_" + str(key) + ".pkl")
        save_pkl(obj=better_than_NE_idx, path=checkpoint_dir + "better_than_NE_idx_" + str(key) + ".pkl")































