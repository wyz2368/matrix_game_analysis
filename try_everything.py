import numpy as np
import os
import datetime
import game_generator
# from psro_trainer import PSRO_trainer
from utils import *
from itertools import product
# from meta_strategies import double_oracle,fictitious_play

a = np.zeros(5)

def add(a):
    a += 4

add(a)
print(a)


