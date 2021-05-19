from meta_strategies import double_oracle, fictitious_play, mrcp_solver, prd_solver, iterative_double_oracle
from meta_strategies import iterated_prd, iterative_double_oracle_player_selection, regret_controled_RD
from game_generator import Game_generator
from psro_trainer import PSRO_trainer
from utils import set_random_seed
from nash_solver.gambit_tools import load_pkl

from absl import app
from absl import flags
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import sys
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_rounds", 10, "The number of rounds starting with different.")
flags.DEFINE_integer("num_strategies", 100, "The number of rounds starting with different.")
flags.DEFINE_integer("num_iterations", 20, "The number of rounds starting with different.")
flags.DEFINE_string("game_type", "zero_sum", "Type of synthetic game.")
flags.DEFINE_integer("seed", None, "The seed to control randomness.")
flags.DEFINE_boolean("MRCP_deterministic", True, "mrcp should return a same value given the same empirical game")
flags.DEFINE_string("closed_method", "alter", "Method for handling closeness of the MRCP")

def psro(generator,
         game_type,
         num_rounds,
         seed,
         checkpoint_dir,
         num_iterations=20,
         closed_method="alter"):
    if game_type == "zero_sum":
        meta_games = generator.zero_sum_game()
    elif game_type == "general_sum":
        meta_games = generator.general_sum_game()
    elif game_type == "symmetric_zero_sum":
        meta_games = generator.symmetric_zero_sum_game()
    elif game_type == "kuhn":
        kuhn_meta_games = load_pkl("./MRCP/kuhn_meta_game.pkl")
        meta_games = kuhn_meta_games[0] # The first element of kuhn_meta_game.pkl is meta_games.
        generator.num_strategies = 64
    else:
        for pkl in os.listdir('efg_game'):
            print(pkl)
            if pkl.split('.pkl')[0] == game_type:
                with open('efg_game/'+pkl,'rb') as f:
                    meta_games = pickle.load(f)
        if not 'meta_games' in locals():
            raise ValueError
    
    # for example 1 in paper
    # meta_games = [np.array([[0,-0.1,-3],[0.1,0,2],[3,-2,0]]),np.array([[0,0.1,3],[-0.1,0,-2],[-3,2,0]])]
    # generator.num_strategies = 3
    # num_rounds = 1
    # num_iterations = 10
    init_strategies = np.random.randint(0, meta_games[0].shape[0], num_rounds)

    DO_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=double_oracle,
                           checkpoint_dir=checkpoint_dir,
                           num_iterations=num_iterations,
                           seed=seed,
                           init_strategies=init_strategies)

    FP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
                           num_rounds=num_rounds,
                           meta_method=fictitious_play,
                           checkpoint_dir=checkpoint_dir,
                           num_iterations=num_iterations,
                           seed=seed,
                           init_strategies=init_strategies)

    PRD_trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=generator.num_strategies,
                               num_rounds=num_rounds,
                               meta_method=prd_solver,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=num_iterations,
                               seed=seed,
                               init_strategies=init_strategies)

    CRD_trainer = PSRO_trainer(meta_games=meta_games,
                               num_strategies=num_strategies,
                               num_rounds=num_rounds,
                               meta_method=regret_controled_RD,
                               checkpoint_dir=checkpoint_dir,
                               num_iterations=num_iterations,
                               seed=seed,
                               init_strategies=init_strategies)

    # IDO_trainer = PSRO_trainer(meta_games=meta_games,
    #                            num_strategies=generator.num_strategies,
    #                            num_rounds=num_rounds,
    #                            meta_method=iterative_double_oracle,
    #                            checkpoint_dir=checkpoint_dir,
    #                            num_iterations=num_iterations,
    #                            seed=seed,
    #                            init_strategies=init_strategies)
    #
    # IPRD_trainer = PSRO_trainer(meta_games=meta_games,
    #                             num_strategies=generator.num_strategies,
    #                             num_rounds=num_rounds,
    #                             meta_method=iterated_prd,
    #                             checkpoint_dir=checkpoint_dir,
    #                             num_iterations=num_iterations,
    #                             seed=seed,
    #                             init_strategies=init_strategies)

    # IDOS_trainer = PSRO_trainer(meta_games=meta_games,
    #                             num_strategies=generator.num_strategies,
    #                             num_rounds=num_rounds,
    #                             meta_method=iterative_double_oracle_player_selection,
    #                             checkpoint_dir=checkpoint_dir,
    #                             num_iterations=num_iterations,
    #                             seed=seed,
    #                             init_strategies=init_strategies)

    MRCP_trainer = PSRO_trainer(meta_games=meta_games,
                           num_strategies=generator.num_strategies,
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
    df.to_csv(checkpoint_dir+game_type+'_DO.csv',index=False)
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

    # PRD_trainer.loop()
    # print("#####################################")
    # print('PRD looper finished looping')
    # print("#####################################")
    # df = pd.DataFrame(np.transpose(PRD_trainer.neconvs + PRD_trainer.mrconvs), \
    #                     columns=nashconv_names + mrconv_names)
    # df.to_csv(checkpoint_dir + game_type + '_PRD.csv', index=False)
    # with open(checkpoint_dir + game_type + '_mrprofile_PRD.pkl', 'wb') as f:
    #     pickle.dump(PRD_trainer.mrprofiles, f)

    CRD_trainer.loop()
    print("#####################################")
    print('CRD looper finished looping')
    print("#####################################")
    df = pd.DataFrame(np.transpose(CRD_trainer.neconvs + CRD_trainer.mrconvs), \
                      columns=nashconv_names + mrconv_names)
    df.to_csv(checkpoint_dir + game_type + '_CRD.csv', index=False)
    with open(checkpoint_dir + game_type + '_mrprofile_CRD.pkl', 'wb') as f:
        pickle.dump(CRD_trainer.mrprofiles, f)

    # IDO_trainer.loop()
    # print("#####################################")
    # print('IDO looper finished looping')
    # print("#####################################")
    # df = pd.DataFrame(np.transpose(IDO_trainer.neconvs + IDO_trainer.mrconvs), \
    #                     columns=nashconv_names + mrconv_names)
    # df.to_csv(checkpoint_dir + game_type + '_IDO.csv', index=False)
    # with open(checkpoint_dir + game_type + '_mrprofile_IDO.pkl', 'wb') as f:
    #     pickle.dump(IDO_trainer.mrprofiles, f)
    #
    # IPRD_trainer.loop()
    # print("#####################################")
    # print('IPRD looper finished looping')
    # print("#####################################")
    # df = pd.DataFrame(np.transpose(IPRD_trainer.neconvs + IPRD_trainer.mrconvs), \
    #                      columns=nashconv_names + mrconv_names)
    # df.to_csv(checkpoint_dir + game_type + '_IPRD.csv', index=False)
    # with open(checkpoint_dir + game_type + '_mrprofile_IPRD.pkl', 'wb') as f:
    #     pickle.dump(IPRD_trainer.mrprofiles, f)

    # IDOS_trainer.loop()
    # print("#####################################")
    # print('IDOS looper finished looping')
    # print("#####################################")
    # df = pd.DataFrame(np.transpose(IDOS_trainer.neconvs + IDOS_trainer.mrconvs), \
    #                     columns=nashconv_names + mrconv_names)
    # df.to_csv(checkpoint_dir + game_type + '_IDOS.csv', index=False)
    # with open(checkpoint_dir + game_type + '_mrprofile_IDOS.pkl', 'wb') as f:
    #     pickle.dump(IDOS_trainer.mrprofiles, f)

    # MRCP_trainer.loop()
    # print("#####################################")
    # print('MRCP looper finished looping')
    # print("#####################################")
    # df = pd.DataFrame(np.transpose(MRCP_trainer.neconvs+MRCP_trainer.mrconvs),\
    #         columns=nashconv_names+mrconv_names)
    # df.to_csv(checkpoint_dir+game_type+'_MRCP.csv',index=False)
    # with open(checkpoint_dir + game_type + '_mrprofile_MRCP.pkl','wb') as f:
    #     pickle.dump(DO_trainer.mrprofiles, f)

    print("The current game type is ", game_type)
    print("DO neco av:", np.mean(DO_trainer.neconvs, axis=0))
    print("DO mrcp av:", np.mean(DO_trainer.mrconvs, axis=0))
    print("FP fpco av:", np.mean(FP_trainer.nashconvs, axis=0))
    print("FP neco av:", np.mean(FP_trainer.neconvs, axis=0))
    print("FP mrcp av:", np.mean(FP_trainer.mrconvs, axis=0))
    # print("PRD prdco av:", np.mean(PRD_trainer.nashconvs, axis=0))
    # print("PRD neco av:", np.mean(PRD_trainer.neconvs, axis=0))
    # print("PRD mrcp av:", np.mean(PRD_trainer.mrconvs, axis=0))
    print("CRD CRDco av:", np.mean(CRD_trainer.nashconvs, axis=0))
    print("CRD neco av:", np.mean(CRD_trainer.neconvs, axis=0))
    print("CRD mrcp av:", np.mean(CRD_trainer.mrconvs, axis=0))
    # print("IDO IDOco av:", np.mean(IDO_trainer.nashconvs, axis=0))
    # print("IDO neco av:", np.mean(IDO_trainer.neconvs, axis=0))
    # print("IDO mrcp av:", np.mean(IDO_trainer.mrconvs, axis=0))
    # print("IPRD IDOco av:", np.mean(IDO_trainer.nashconvs, axis=0))
    # print("IDO neco av:", np.mean(IDO_trainer.neconvs, axis=0))
    # print("IDO mrcp av:", np.mean(IDO_trainer.mrconvs, axis=0))
    # print("IDOS IDOSco av:", np.mean(IDOS_trainer.nashconvs, axis=0))
    # print("IDOS neco av:", np.mean(IDOS_trainer.neconvs, axis=0))
    # print("IDOS mrcp av:", np.mean(IDOS_trainer.mrconvs, axis=0))
    # print("MR neco av:", np.mean(MRCP_trainer.neconvs, axis=0))
    # print("MR mrcp av:", np.mean(MRCP_trainer.mrconvs, axis=0))

    print("====================================================")
    

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    seed = set_random_seed(FLAGS.seed)
    if not FLAGS.MRCP_deterministic:
        seed = None # invalidate the seed so it does not get passed into psro_trainer

    # root_path = './' + FLAGS.game_type + "_se_" + '/'
    #
    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)

    generator = Game_generator(FLAGS.num_strategies)
    checkpoint_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_se_'+ FLAGS.game_type + "_" + str(seed)
    checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

    # game_list = ["zero_sum", "general_sum"]

    psro(generator=generator,
         game_type=FLAGS.game_type,
         num_rounds=FLAGS.num_rounds,
         seed=seed,
         checkpoint_dir=checkpoint_dir,
         num_iterations=FLAGS.num_iterations,
         closed_method=FLAGS.closed_method)


if __name__ == "__main__":
  app.run(main)
