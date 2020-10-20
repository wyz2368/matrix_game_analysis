from nash_solver.gambit_tools import load_pkl
import numpy as np

root_path = './data/'
improvement = load_pkl(root_path + "performance_improvement_10.pkl")
regret = load_pkl(root_path + "regret_of_samples_10.pkl")

cnt = 0
for i in improvement:
    if np.abs(i) < 1e-5:
        cnt += 1

print("improvement:", improvement[:20])
print("regret:", regret[:20])
print("cnt:", cnt)