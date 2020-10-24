import numpy as np
from utils import set_random_seed
from meta_strategies import double_oracle
from MRCP.minimum_regret_profile import minimum_regret_profile_calculator

class PSRO_trainer(object):
    def __init__(self,
                 meta_games,
                 num_strategies,
                 num_rounds,
                 meta_method,
                 checkpoint_dir,
                 num_iterations=20,
                 seed=None,
                 empricial_game_record=None,
                 calculate_neconv=True,
                 calculate_mrcpconv=True,
                 init_strategies=None,
                 closed_method="alter"):
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
        self.num_rounds = num_rounds
        self.meta_method = meta_method
        self.num_strategies = num_strategies
        self.checkpoint_dir = checkpoint_dir
        self.mrcp_calculator = minimum_regret_profile_calculator(full_game=meta_games)
        self.seed = seed
        self.calculate_neconv = calculate_neconv
        self.calculate_mrconv = calculate_mrcpconv
        if init_strategies is not None:
            assert isinstance(init_strategies, int) or len(init_strategies) == num_rounds, \
                    "provide initial strategies with right length"
            self.init_strategies = np.array([init_strategies for _ in range(num_rounds)],dtype=int) if isinstance(init_strategies,int) else np.array(init_strategies,dtype=int)
        else:
            self.init_strategies = np.random.randint(0, num_strategies, num_rounds)

        self.empirical_games = [[], []]
        self.num_iterations = num_iterations

        self.nashconvs = []  # Heuristic-conv. The name is retained for convenience sake
        self.neconvs = []
        self.mrconvs = []
        self.mrprofiles = []

        # Record empirical game.
        self.empricial_game_record = empricial_game_record
        if empricial_game_record is not None:
            self.empirical_games_dict = {}

        self.closed_method = closed_method


    def init_round(self, init_strategy):
        #init_strategy = np.random.randint(0, self.num_strategies)
        #init_strategy = 0
        self.empirical_games = [[init_strategy], [init_strategy]]

    def iteration(self):
        nashconv_list = []
        neconv_list = []
        mrconv_list = []
        mrprofile_list = []

        # Tricky Detail: mrcp does not calculate NE's first empirical game's mrcp value
        # ne does not calculate mrcp's first empirical game's NE-based regret
        if self.calculate_mrconv:
            if self.meta_method.__name__ != 'mrcp_solver':
                mrcp_profile, mrcp_value = self.mrcp_calculator(self.empirical_games)
                mrconv_list.append(mrcp_value)
                mrprofile_list.append(mrcp_profile)

        if self.meta_method.__name__ != 'double_oracle':
            _, neconv = double_oracle(self.meta_games,self.empirical_games,self.checkpoint_dir)
            neconv_list.append(neconv)

        for it in range(self.num_iterations):
            print('################## Iteration {} ###################'.format(it))
            dev_strs, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
            nashconv_list.append(nashconv)

            self.empirical_games[0].append(dev_strs[0])
            self.empirical_games[0] = sorted(self.empirical_games[0])
            self.empirical_games[1].append(dev_strs[1])
            self.empirical_games[1] = sorted(self.empirical_games[1])

            if self.empricial_game_record is not None and it in self.empricial_game_record:
                self.empirical_games_dict[it] = self.empirical_games.copy()

            if self.calculate_neconv:
                if self.meta_method.__name__ != 'double_oracle':
                    _, neconv = double_oracle(self.meta_games,
                                              self.empirical_games,
                                              self.checkpoint_dir)
                    neconv_list.append(neconv)
                else:
                    neconv_list.append(nashconv) 

            if self.calculate_mrconv:
                if self.meta_method.__name__ != 'mrcp_solver':
                    mrcp_profile, mrcp_value = self.mrcp_calculator(self.empirical_games)
                    mrconv_list.append(mrcp_value)
                    mrprofile_list.append(mrcp_profile)
                else:
                    mrconv_list.append(nashconv)
                    mrprofile_list.append(self.meta_method.mrcp_calculator.mrcp_profile)
        
        # Tricky part: Nashconv does not add the last value after update
        # mrcp does not add its last own value after its last update
        # NE does not add its last own value after its last update
        if self.meta_method.__name__ == 'mrcp_solver':
            _, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir, method=self.closed_method)
        else:
            _, nashconv = self.meta_method(self.meta_games, self.empirical_games, self.checkpoint_dir)
        nashconv_list.append(nashconv)
        if self.meta_method.__name__ == 'mrcp_solver':
            mrconv_list.append(nashconv)
            mrprofile_list.append(self.meta_method.mrcp_calculator.mrcp_profile)
        if self.meta_method.__name__ == 'double_oracle':
            neconv_list.append(nashconv)
        
        self.nashconvs.append(nashconv_list)
        self.mrconvs.append(mrconv_list)
        self.mrprofiles.append(mrprofile_list)
        self.neconvs.append(neconv_list)

    def loop(self):
        for i in range(self.num_rounds):
            self.init_round(self.init_strategies[i])
            # reset to same random seed to guarantee MRCP
            # being deterministic given empirical game
            if self.seed is not None:
                set_random_seed(self.seed) 

            self.iteration()
            self.mrcp_calculator.clear()

    def get_empirical_game(self):
        return self.empirical_games

    def get_recorded_empirical_game(self):
        return self.empirical_games_dict
