# scripts/debug_alphabeta.py
import numpy as np
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves
from agents.alphabeta_agent import select_move_timed, evaluate

# Test 1: what does evaluate() return for the opening position?
game = reversi_game()
print("Opening position evaluation:")
print(f"  From white perspective: {evaluate(game.board, 1):.2f}")
print(f"  From black perspective: {evaluate(game.board, -1):.2f}")

# Test 2: play 5 games with depth 1 vs depth 1 and print every move
print("\nDepth 1 vs Depth 1 — 3 games:")
for game_num in range(3):
    game = reversi_game()
    consecutive_passes = 0
    moves = 0

    while True:
        player = game.turn
        legal  = get_legal_moves(game.board, player)

        if not legal:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.turn = -game.turn
            continue
        else:
            consecutive_passes = 0

        x, y = select_move_timed(
            game.board, player,
            time_limit = 5.0,
            max_depth  = 1
        )

        if x == -1 and y == -1:
            game.turn = -game.turn
            continue

        game.step(x, y, player, commit=True)
        game.turn = -game.turn
        moves += 1

    white = int(np.sum(game.board == 1))
    black = int(np.sum(game.board == -1))
    winner = "WHITE" if white > black else "BLACK" if black > white else "DRAW"
    print(f"  Game {game_num+1}: W={white} B={black} → {winner} wins "
          f"({moves} moves)")

# Test 3: check if evaluate is correctly signed
print("\nEvaluate sign check:")
game = reversi_game()
# Manually create a position where white has corners
game.board = np.zeros((8,8))
game.board[0,0] = 1   # white corner
game.board[0,7] = 1   # white corner  
game.board[7,0] = -1  # black corner
game.board[3,3] = 1
game.board[4,4] = 1
game.board[3,4] = -1
game.board[4,3] = -1

print(f"  White has 2 corners, black has 1:")
print(f"  evaluate(board, white=1):  {evaluate(game.board, 1):.2f} "
      f"(should be POSITIVE)")
print(f"  evaluate(board, black=-1): {evaluate(game.board, -1):.2f} "
      f"(should be NEGATIVE)")