# scripts/debug_timed.py
import time
import numpy as np
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves
from agents.alphabeta_agent import select_move_timed, select_move

game = reversi_game()

print("Comparing timed vs untimed at depth 3:")
print(f"Legal moves: {get_legal_moves(game.board, 1)}")

# Untimed depth 3
start = time.time()
x1, y1 = select_move(game.board, 1, depth=3)
t1 = time.time() - start
print(f"  Untimed d3: ({x1},{y1}) in {t1:.3f}s")

# Timed depth 3 with 1 second
start = time.time()
x2, y2 = select_move_timed(game.board, 1, time_limit=1.0, max_depth=3)
t2 = time.time() - start
print(f"  Timed d3 (1s): ({x2},{y2}) in {t2:.3f}s")

# Timed depth 3 with 5 seconds
start = time.time()
x3, y3 = select_move_timed(game.board, 1, time_limit=5.0, max_depth=3)
t3 = time.time() - start
print(f"  Timed d3 (5s): ({x3},{y3}) in {t3:.3f}s")

print("\nPlaying 3 games with 1s limit:")
for i in range(3):
    g = reversi_game()
    cp = 0
    while True:
        player = g.turn
        legal = get_legal_moves(g.board, player)
        if not legal:
            cp += 1
            if cp >= 2: break
            g.turn = -g.turn
            continue
        else:
            cp = 0
        x, y = select_move_timed(g.board, player, time_limit=1.0, max_depth=3)
        if x == -1 and y == -1:
            g.turn = -g.turn
            continue
        g.step(x, y, player, commit=True)
        g.turn = -g.turn
    w = int(np.sum(g.board==1))
    b = int(np.sum(g.board==-1))
    print(f"  Game {i+1}: W={w} B={b} → {'WHITE' if w>b else 'BLACK'} wins")

print("\nPlaying 3 games with 5s limit:")
for i in range(3):
    g = reversi_game()
    cp = 0
    while True:
        player = g.turn
        legal = get_legal_moves(g.board, player)
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
    w = int(np.sum(g.board==1))
    b = int(np.sum(g.board==-1))
    print(f"  Game {i+1}: W={w} B={b} → {'WHITE' if w>b else 'BLACK'} wins")