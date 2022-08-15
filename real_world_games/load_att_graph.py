import pickle
from nash_solver.gambit_tools import load_pkl, save_pkl
import numpy as np



with open('./att_graph_game.txt', 'r') as file:
    data = file.read()

data = data.split(' ')
defender_payoffs = data[0::2][:-1]
attacker_payoffs = data[1::2]

for i in range(len(defender_payoffs)):
    defender_payoffs[i] = np.round(float(defender_payoffs[i]), 2)
    attacker_payoffs[i] = np.round(float(attacker_payoffs[i]), 2)

# print(data[:10])
# print(defender_payoffs[:5], np.shape(defender_payoffs))
# print(attacker_payoffs[:5], np.shape(attacker_payoffs))

defender_payoffs_noisy = defender_payoffs + np.random.randint(low=-100, high=0, size=len(defender_payoffs))
attacker_payoffs_noisy = attacker_payoffs + np.random.randint(low=-100, high=100, size=len(attacker_payoffs))


defender_payoffs = np.reshape(defender_payoffs, (47, 47))
attacker_payoffs = np.reshape(attacker_payoffs, (47, 47))

defender_payoffs_noisy = np.reshape(defender_payoffs_noisy, (47, 47))
attacker_payoffs_noisy = np.reshape(attacker_payoffs_noisy, (47, 47))

# print(defender_payoffs[0, :])
# print(attacker_payoffs[0, :])


meta_game_noisy = []
meta_game_noisy.append(defender_payoffs_noisy)
meta_game_noisy.append(attacker_payoffs_noisy)


meta_game = []
meta_game.append(defender_payoffs)
meta_game.append(attacker_payoffs)

meta_game = np.array(meta_game)
meta_game_noisy = np.array(meta_game_noisy)

# print(meta_game[0])
# print(meta_game[1])
# np.savetxt("./attack_graph_payoff_matrix0.txt", meta_game[0])
# np.savetxt("./attack_graph_payoff_matrix1.txt", meta_game[1])

save_pkl(meta_game, './att_graph.pkl')
#### save_pkl(meta_game_noisy, './att_graph_noisy.pkl')




# From Mason

# data = [
#   [65.44009891412475, -228.40326067409038 ],
#   [111.81016373573755, -234.95706659467643 ],
#   [109.44662340751165, -231.4451634056161 ],
#   [88.94720071337322, -218.1839159286827 ],
#   [75.21491512623307, -209.77744472023267 ],
#   [81.615419634155, -213.38062152827268 ],
#   [13.691437815788827, -77.087626349741 ],
#   [27.996686842538523, -83.09281879117668 ],
#   [31.259191631832465, -83.2989863508063 ],
#   [52.85284738633463, -90.37091820189133 ],
#   [61.29275252034736, -93.05685244712781 ],
#   [53.84774777939639, -89.52765756402921 ],
#   [19.883987594318704, -79.22463781543983 ],
#   [0.977460459520359, -61.076006496183 ],
#   [5.878211030565649, -63.64733410994303 ],
#   [69.11120875201564, -101.16079242708132 ],
#   [38.24401497221616, -75.81831336164885 ],
#   [47.29379113730121, -82.34248465660843 ],
#   [13.79718036023141, -73.79817318726614 ],
#   [13.458082792228732, -70.99891411950183 ],
#   [15.957527435494448, -65.12880759863889 ],
#   [69.52770041244072, -96.19198715517936 ],
#   [44.22647536429126, -72.59049163114068 ],
#   [60.56613028067599, -86.90306204922337 ],
#   [22.938166285592043, -83.72424895108888 ],
#   [24.674915336093104, -85.9845293365598 ],
#   [4.473415896980356, -49.34753610569406 ],
#   [15.378832605348164, -63.454194778280474 ],
#   [56.3489835059106, -94.75714096387965 ],
#   [63.26632464289011, -101.11693707984882 ],
#   [25.310957809335587, -83.18610234253374 ],
#   [24.458984225262657, -85.28225687619769 ],
#   [20.410551302043203, -76.84560962431138 ],
#   [19.69015365817591, -65.15558268037755 ],
#   [11.240005469478922, -41.596742626718914 ],
#   [64.89052240134104, -86.55120870619164 ]
# ]