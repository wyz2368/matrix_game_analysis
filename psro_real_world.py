from meta_strategies import double_oracle, fictitious_play, mrcp_solver
from psro_trainer import PSRO_trainer
from utils import set_random_seed
from nash_solver.gambit_tools import load_pkl

from absl import app
from absl import flags
import os
import pickle
import numpy as np
import pandas as pd
import sys
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "The number of rounds starting with different.")
flags.DEFINE_string("game_type", "real_world", "Type of synthetic game.")
flags.DEFINE_integer("seed", None, "The seed to control randomness.")
flags.DEFINE_boolean("MRCP_deterministic", True, "mrcp should return a same value given the same empirical game")
flags.DEFINE_string("closed_method", "alter", "Method for handling closeness of the MRCP")

def psro(meta_games,
         game_type,
         num_rounds,
         seed,
         checkpoint_dir,
         num_iterations=20,
         closed_method="alter"):

    num_strategies = meta_games[0].shape[0]
    init_strategies = 0

    DO_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=num_strategies,
                           num_rounds=num_rounds,
                           meta_method=double_oracle,
                           checkpoint_dir=checkpoint_dir,
                           num_iterations=num_iterations,
                           seed=seed,
                           init_strategies=init_strategies)

    FP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=num_strategies,
                           num_rounds=num_rounds,
                           meta_method=fictitious_play,
                           checkpoint_dir=checkpoint_dir,
                           num_iterations=num_iterations,
                           seed=seed,
                           init_strategies=init_strategies)

    MRCP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=num_strategies,
                           num_rounds=num_rounds,
                           meta_method=mrcp_solver,
                           checkpoint_dir=checkpoint_dir,
                           num_iterations=num_iterations,
                           seed=seed,
                           init_strategies=init_strategies,
                           closed_method=closed_method)


    DO_trainer.loop()
    print("#####################################")
    print('DO looper finished looping')
    print("#####################################")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + game_type + '_meta_games.pkl','wb') as f:
        pickle.dump(meta_games, f)
    nashconv_names = ['nashconvs_'+str(t) for t in range(len(DO_trainer.neconvs))]
    mrconv_names = ['mrcpcons_'+str(t) for t in range(len(DO_trainer.mrconvs))]
    df = pd.DataFrame(np.transpose(DO_trainer.neconvs+DO_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir + game_type +'_DO.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_DO.pkl','wb') as f:
        pickle.dump(DO_trainer.mrprofiles, f)

    FP_trainer.loop()
    print("#####################################")
    print('FP looper finished looping')
    print("#####################################")
    df = pd.DataFrame(np.transpose(FP_trainer.neconvs+FP_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir+game_type+'_FP.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_FP.pkl','wb') as f:
        pickle.dump(FP_trainer.mrprofiles, f)

    MRCP_trainer.loop()
    print("#####################################")
    print('MRCP looper finished looping')
    print("#####################################")
    df = pd.DataFrame(np.transpose(MRCP_trainer.neconvs+MRCP_trainer.mrconvs),\
            columns=nashconv_names+mrconv_names)
    df.to_csv(checkpoint_dir+game_type+'_MRCP.csv',index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_MRCP.pkl','wb') as f:
        pickle.dump(DO_trainer.mrprofiles, f)

    print("The current game type is ", game_type)
    print("DO neco av:", np.mean(DO_trainer.neconvs, axis=0))
    print("DO mrcp av:", np.mean(DO_trainer.mrconvs, axis=0))
    print("FP fpco av:", np.mean(FP_trainer.nashconvs, axis=0))
    print("FP neco av:", np.mean(FP_trainer.neconvs, axis=0))
    print("FP mrcp av:", np.mean(FP_trainer.mrconvs, axis=0))
    print("MR neco av:", np.mean(MRCP_trainer.neconvs, axis=0))
    print("MR mrcp av:", np.mean(MRCP_trainer.mrconvs, axis=0))

    print("====================================================")
    

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    seed = set_random_seed(FLAGS.seed)
    if not FLAGS.MRCP_deterministic:
        seed = None # invalidate the seed so it does not get passed into psro_trainer

    root_path = './' + FLAGS.game_type + "_se" + '/'

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    real_world_meta_games = load_pkl('./real_world_games/real_world_meta_games.pkl')

    for game_type in real_world_meta_games:
        print("================================================")
        print("======The current game is ", game_type, "=========")
        print("================================================")

        checkpoint_dir = game_type + "_" + str(seed)
        checkpoint_dir = os.path.join(os.getcwd(), root_path, checkpoint_dir) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

        psro(meta_games=real_world_meta_games[game_type],
             game_type=game_type,
             num_rounds=1,
             seed=seed,
             checkpoint_dir=checkpoint_dir,
             num_iterations=FLAGS.num_iterations,
             closed_method=FLAGS.closed_method)


if __name__ == "__main__":
  app.run(main)