"""
A python scripts_matrix for generating all batch files for parameter tuning
"""
import shutil
import os
import itertools
import numpy as np
import itertools
import copy


ORIGIN = './base_slurm.sh'
BASE = 'python psro_att_graph.py --num_iterations=15 --closed_method=dev --game_type=noisy'
BALANCE = '--balance_factor='
MINUS = '--minus='


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

def write_line(file, line):
    file.write(line)

def bash_factory():
    factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mode = [True, False]

    params = list(itertools.product(factors, mode)) #TODO: change is_max_other_util

    for i, param in enumerate(params):
        f, m = param
        target = './scripts_matrix/z_' + str(i) + '.sh'
        copy_file(ORIGIN, target)

        with open(target, 'a') as file:
            commands = [BASE, BALANCE + str(f), MINUS + str(m)]

            new_command = " ".join(commands)
            write_line(file, '\n')
            write_line(file, new_command)


if __name__ == '__main__':
    bash_factory()





