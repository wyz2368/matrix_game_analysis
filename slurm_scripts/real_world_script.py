"""
A python scripts for generating all batch files for parameter tuning
"""
import shutil
import os
import numpy as np
import itertools
import copy


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")


def copy_file(original, target):
    shutil.copyfile(original, target)
    # print("Copy file from: ", original)
    # print("Copy file to: ", target)


def delete_last_line(file):
    # Move the pointer (similar to a cursor in a text editor) to the end of the file
    file.seek(0, os.SEEK_END)

    # This code means the following code skips the very last character in the file -
    # i.e. in the case the last line is null we delete the last line
    # and the penultimate one
    pos = file.tell() - 1

    # Read each character in the file one at a time from the penultimate
    # character going backwards, searching for a newline character
    # If we find a new line, exit the search
    while pos > 0 and file.read(1) != "\n":
        pos -= 1
        file.seek(pos, os.SEEK_SET)

    # So long as we're not at the start of the file, delete all the characters ahead
    # of this position
    if pos > 0:
        file.seek(pos, os.SEEK_SET)
        file.truncate()


def write_line(file, line):
    file.write(line)


# TARGET_DIR = os.getcwd() + '/slurm_scripts/'
ORIGIN = os.path.dirname(os.path.realpath(__file__)) + '/base_slurm.sh'
MODULE1 = "module load python3.6-anaconda/5.2.0"
MODULE2 = "cd $(dirname $(dirname '${SLURM_SUBMIT_DIR}'))"
OUTPUT = "#SBATCH --output="
# dqn and ars switching, does not swap heuristics
# COMMAND = "python ../se_example.py --game_name=leduc_poker --quiesce=False --gpsro_iterations=200 --sbatch_run=True --log_train=False --number_training_episodes=10000 --number_training_episodes_ars=300000 --root_result_folder=root_result_ars_dqn --heuristic_list=general_nash_strategy --fast_oracle_period=3 --slow_oracle_period=1 --switch_fast_slow=True"
# switch heuristic without switching oracle in fixed pattern
COMMAND = "python ../se_example.py --game_name=leduc_poker --quiesce=False --gpsro_iterations=150 --sbatch_run=True --log_train=False --number_training_episodes=10000 --root_result_folder=root_result_uni_refute --heuristic_list=uniform_strategy,general_nash_strategy --switch_fast_slow=False --switch_heuristic_regardless_of_oracle=True"


def bash_factory(dir_name='scripts_uni_refute', num_files=10, grid_search_flag=True):
    bash_path = os.path.dirname(os.path.realpath(__file__)) + '/' + dir_name + '/'
    if os.path.exists(bash_path):
        shutil.rmtree(bash_path, ignore_errors=True)
    else:
        mkdir(bash_path)
    output_path = os.path.dirname(os.path.realpath(__file__)) + '/' + dir_name + '_output' + '/'
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    else:
        mkdir(output_path)

    param_dict = {'seed': [np.random.randint(low=0, high=1e5) for _ in range(10)]}

    if grid_search_flag:
        params = grid_search(param_dict, search_ars_bd=False)
    else:
        # provides param dict keys, and params in iterrable form
        param_dict = {'num_directions': [],
                      'num_best_directions': [],
                      'ars_learning_rate': [],
                      'noise': [],
                      'seed': [],
                      'number_training_episodes': []}
        _params = [[16, 16, 0.01, 0.3, 17027],
                   [16, 16, 0.01, 0.5, 46327],
                   [16, 16, 0.03, 0.07, 67519],
                   [16, 16, 0.03, 0.5, 31871],
                   [40, 20, 0.03, 0.5, 9487],
                   [40, 40, 0.07, 0.07, 26441],
                   [80, 20, 0.07, 0.07, 50603],
                   [80, 40, 0.07, 0.07, 44905],
                   [20, 20, 0.07, 0.3, 49510],
                   [40, 20, 0.07, 0.3, 73876],
                   [40, 40, 0.07, 0.3, 56240],
                   [80, 20, 0.07, 0.5, 45147],
                   [80, 40, 0.1, 0.07, 45999],
                   [80, 20, 0.1, 0.1, 35121],
                   [80, 20, 0.1, 0.3, 77496],
                   [16, 16, 0.3, 0.1, 38457],
                   [80, 20, 0.3, 0.5, 65210]]
        params = []
        episodes = [100000, 300000, 500000]
        for ele in _params:
            for epi in episodes:
                params.append(tuple(ele + [epi]))

    for i, item in enumerate(params):
        nick_name = ''
        for value in item:
            nick_name += "_" + str(value)
        target = bash_path + str(i) + nick_name + '.sh'
        copy_file(ORIGIN, target)
        new_command = copy.copy(COMMAND)
        for j, key in enumerate(param_dict.keys()):
            arg = ' --' + key + ' ' + str(item[j])
            new_command += arg
        with open(target, 'a') as file:
            write_line(file, OUTPUT + output_path + str(i) + nick_name + '.log' + '\n')
            write_line(file, MODULE1 + '\n')
            write_line(file, MODULE2 + '\n')
            write_line(file, new_command)


if __name__ == '__main__':
    bash_factory()

