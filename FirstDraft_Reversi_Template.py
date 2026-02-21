# reversi_battle.py
# Run: python reversi_battle.py
# Python 3.9+

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import random
import time



EMPTY = "."
BLACK = "B"
WHITE = "W"

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1),
]

# Classic positional weights (corners great, X-squares terrible, etc.)
WEIGHTS = [
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
]

@dataclass(frozen=True)
class Move:
    r: int
    c: int

def opponent(player: str) -> str:
    return BLACK if player == WHITE else WHITE

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8

def new_board() -> List[List[str]]:
    b = [[EMPTY for _ in range(8)] for _ in range(8)]
    # Standard start
    b[3][3] = WHITE
    b[3][4] = BLACK
    b[4][3] = BLACK
    b[4][4] = WHITE
    return b

def print_board(board: List[List[str]]) -> None:
    print("  " + " ".join(str(i) for i in range(8)))
    for r in range(8):
        print(str(r) + " " + " ".join(board[r]))
    bcount, wcount = count_discs(board)
    print(f"Score -> B: {bcount}  W: {wcount}")

def count_discs(board: List[List[str]]) -> Tuple[int, int]:
    b = sum(cell == BLACK for row in board for cell in row)
    w = sum(cell == WHITE for row in board for cell in row)
    return b, w

def discs_to_flip(board: List[List[str]], player: str, move: Move) -> List[Tuple[int, int]]:
    """Return list of coordinates that would be flipped if player plays move."""
    if not in_bounds(move.r, move.c) or board[move.r][move.c] != EMPTY:
        return []
    flips: List[Tuple[int, int]] = []
    opp = opponent(player)

    for dr, dc in DIRECTIONS:
        r, c = move.r + dr, move.c + dc
        line: List[Tuple[int, int]] = []
        while in_bounds(r, c) and board[r][c] == opp:
            line.append((r, c))
            r += dr
            c += dc
        if in_bounds(r, c) and board[r][c] == player and line:
            flips.extend(line)

    return flips

def legal_moves(board: List[List[str]], player: str) -> List[Move]:
    moves: List[Move] = []
    for r in range(8):
        for c in range(8):
            if board[r][c] == EMPTY and discs_to_flip(board, player, Move(r, c)):
                moves.append(Move(r, c))
    return moves

def apply_move(board: List[List[str]], player: str, move: Move) -> List[List[str]]:
    flips = discs_to_flip(board, player, move)
    if not flips:
        raise ValueError("Illegal move")
    newb = [row[:] for row in board]
    newb[move.r][move.c] = player
    for r, c in flips:
        newb[r][c] = player
    return newb

def game_over(board: List[List[str]]) -> bool:
    return not legal_moves(board, BLACK) and not legal_moves(board, WHITE)

# ---------- AIs ----------

def greedy_ai(board: List[List[str]], player: str) -> Optional[Move]:
    """Simple AI: pick move that flips the most discs; tie-break randomly."""
    moves = legal_moves(board, player)
    if not moves:
        return None
    scored = []
    for m in moves:
        scored.append((len(discs_to_flip(board, player, m)), m))
    best = max(scored, key=lambda x: x[0])[0]
    best_moves = [m for s, m in scored if s == best]
    return random.choice(best_moves)

def evaluate(board: List[List[str]], player: str) -> int:
    """Heuristic evaluation from 'player' perspective."""
    opp = opponent(player)

    # Positional weights
    pos = 0
    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                pos += WEIGHTS[r][c]
            elif board[r][c] == opp:
                pos -= WEIGHTS[r][c]

    # Mobility (having moves is good)
    mob = len(legal_moves(board, player)) - len(legal_moves(board, opp))

    # Disc difference (more important late game; here lightly weighted)
    bcount, wcount = count_discs(board)
    discdiff = (bcount - wcount) if player == BLACK else (wcount - bcount)

    return pos + 8 * mob + 1 * discdiff

def minimax_ai(board: List[List[str]], player: str, depth: int = 4) -> Optional[Move]:
    """Stronger AI: minimax with alpha-beta pruning."""
    moves = legal_moves(board, player)
    if not moves:
        return None

    # Move ordering: try corners/strong squares first (helps alpha-beta)
    def move_key(m: Move) -> int:
        return WEIGHTS[m.r][m.c]
    moves_sorted = sorted(moves, key=move_key, reverse=True)

    best_move = moves_sorted[0]
    best_val = -math.inf
    alpha = -math.inf
    beta = math.inf

    for m in moves_sorted:
        newb = apply_move(board, player, m)
        val = _min_value(newb, player, opponent(player), depth - 1, alpha, beta)
        if val > best_val:
            best_val = val
            best_move = m
        alpha = max(alpha, best_val)

    return best_move

def _max_value(board: List[List[str]], root_player: str, to_move: str, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or game_over(board):
        return evaluate(board, root_player)

    moves = legal_moves(board, to_move)
    if not moves:
        # pass turn
        return _min_value(board, root_player, opponent(to_move), depth - 1, alpha, beta)

    moves = sorted(moves, key=lambda m: WEIGHTS[m.r][m.c], reverse=True)
    v = -math.inf
    for m in moves:
        v = max(v, _min_value(apply_move(board, to_move, m), root_player, opponent(to_move), depth - 1, alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v

def _min_value(board: List[List[str]], root_player: str, to_move: str, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or game_over(board):
        return evaluate(board, root_player)

    moves = legal_moves(board, to_move)
    if not moves:
        # pass turn
        return _max_value(board, root_player, opponent(to_move), depth - 1, alpha, beta)

    moves = sorted(moves, key=lambda m: WEIGHTS[m.r][m.c], reverse=True)
    v = math.inf
    for m in moves:
        v = min(v, _max_value(apply_move(board, to_move, m), root_player, opponent(to_move), depth - 1, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v

# ---------- Battle harness ----------

def battle(ai_black, ai_white, games: int = 1, verbose: bool = True) -> Dict[str, int]:
    """
    ai_black/ai_white: functions(board, player)->Move|None
    Returns win counts.
    """
    results = {"B": 0, "W": 0, "D": 0}

    for g in range(1, games + 1):
        board = new_board()
        to_move = BLACK
        passes = 0

        if verbose:
            print("\n" + "=" * 40)
            print(f"Game {g}")
            print_board(board)

        while not game_over(board):
            moves = legal_moves(board, to_move)
            if not moves:
                passes += 1
                if verbose:
                    print(f"{to_move} has no legal moves -> PASS")
                to_move = opponent(to_move)
                continue

            passes = 0
            if to_move == BLACK:
                move = ai_black(board, BLACK)
            else:
                move = ai_white(board, WHITE)

            if move is None or move not in moves:
                # If an AI returns invalid, fall back to a random legal move
                move = random.choice(moves)

            board = apply_move(board, to_move, move)

            if verbose:
                print(f"{to_move} plays ({move.r},{move.c})")
                print_board(board)

            to_move = opponent(to_move)

        bcount, wcount = count_discs(board)
        if bcount > wcount:
            results["B"] += 1
            if verbose:
                print("BLACK wins!")
        elif wcount > bcount:
            results["W"] += 1
            if verbose:
                print("WHITE wins!")
        else:
            results["D"] += 1
            if verbose:
                print("Draw!")

    return results

if __name__ == "__main__":
    random.seed(5)

    # Example: Greedy (Black) vs Minimax (White)
    # Increase depth to 5 for stronger AI (slower).
    def minimax_depth4(board, player):
        return minimax_ai(board, player, depth=4)

    results = battle(greedy_ai, minimax_depth4, games=1, verbose=True)
    print("\nFinal results:", results)