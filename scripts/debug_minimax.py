# scripts/debug_minimax.py
import numpy as np
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves, apply_move
from agents.alphabeta_agent import evaluate, order_moves

def alphabeta_debug(board, player, depth, alpha, beta, 
                    maximizing, root_player, indent=0):
    legal = get_legal_moves(board, player)
    prefix = "  " * indent

    if depth == 0 or not legal:
        score = evaluate(board, root_player)
        print(f"{prefix}LEAF: player={player} "
              f"root={root_player} score={score:.1f} "
              f"max={maximizing}")
        return score

    ordered = order_moves(legal)[:3]  # only first 3 for readability

    if maximizing:
        value = -np.inf
        for x, y in ordered:
            child = apply_move(board, x, y, player)
            print(f"{prefix}MAX player={player} tries ({x},{y})")
            v = alphabeta_debug(child, -player, depth-1,
                                alpha, beta, False, root_player, indent+1)
            value = max(value, v)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.inf
        for x, y in ordered:
            child = apply_move(board, x, y, player)
            print(f"{prefix}MIN player={player} tries ({x},{y})")
            v = alphabeta_debug(child, -player, depth-1,
                                alpha, beta, True, root_player, indent+1)
            value = min(value, v)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

game = reversi_game()
print("=== White (player=1) searching at depth 2 ===")
print("Expected: white maximizes, black minimizes\n")

legal = get_legal_moves(game.board, 1)
for x, y in legal[:2]:
    child = apply_move(game.board, x, y, 1)
    print(f"Root white tries ({x},{y}):")
    score = alphabeta_debug(child, -1, 1, -np.inf, np.inf,
                            False, 1, indent=1)
    print(f"Score for ({x},{y}): {score:.1f}\n")