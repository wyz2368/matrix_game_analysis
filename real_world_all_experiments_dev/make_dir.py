import os

game_types = ['10,4-Blotto', 'AlphaStar', 'Random_game_of_skill', 'Transitive game',
              'connect_four', 'quoridor(board_size=4)', 'misere(game=tic_tac_toe())', 'hex(board_size=3)',
              'go(board_size=4,komi=6.5)']

for game in game_types:
    os.makedirs("./data/" + game)