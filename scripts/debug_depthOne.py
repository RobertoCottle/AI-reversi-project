# Quick check
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves
from agents.alphabeta_agent import select_move_timed

game = reversi_game()
wins = {'white': 0, 'black': 0}

for _ in range(10):
    g = reversi_game()
    cp = 0
    while True:
        player = g.turn
        legal  = get_legal_moves(g.board, player)
        if not legal:
            cp += 1
            if cp >= 2: break
            g.turn = -g.turn
            continue
        else:
            cp = 0
        x, y = select_move_timed(g.board, player, time_limit=5.0, max_depth=3)
        if x == -1 and y == -1:
            g.turn = -g.turn
            continue
        g.step(x, y, player, commit=True)
        g.turn = -g.turn
    w = int(sum(g.board.flatten() == 1))
    b = int(sum(g.board.flatten() == -1))
    if w > b: wins['white'] += 1
    elif b > w: wins['black'] += 1

print(f"White wins: {wins['white']}/10")
print(f"Black wins: {wins['black']}/10")