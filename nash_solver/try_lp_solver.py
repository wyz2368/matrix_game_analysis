from nash_solver.lp_solver import lp_solve
import numpy as np

meta_games = [np.array([[1 ,0],
                        [0, -1]]),
              np.array([[-1, 0],
                        [0, 1]])
              ]

ne = lp_solve(meta_games)
print(ne)