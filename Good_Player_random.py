# Strong Opponent
# Hybrid Evaluation + Minimax (alpha-beta)
# Generally, the other program was better due to Dynamic Depth (TT used for efficiency [cost memory])

import socket
import pickle
import numpy as np
from reversi import reversi
import random

HOST = "127.0.0.1"
PORT = 33333

# Search depth is set to 3, but we can change it and weigh results.
DEPTH = 3

# Positional Strategy Board - weighted positioning
POS_WEIGHTS = np.array([
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
    [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
    [  5,  -2,  -1,  -1,  -1,  -1,  -2,   5],
    [ 10,  -2,  -1,  -1,  -1,  -1,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100]
], dtype=np.int32)

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
NEIGHBORS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8


def clone_game_from_board(board: np.ndarray) -> reversi:
    g = reversi()
    g.board = board.copy()
    g.white_count = int(np.count_nonzero(g.board == 1))
    g.black_count = int(np.count_nonzero(g.board == -1))
    return g


def valid_moves(g: reversi, turn: int):
    moves = []
    for i in range(8):
        for j in range(8):
            flips = g.step(i, j, turn, commit=False)
            if flips > 0:
                moves.append((i, j))
    return moves


def count_frontier(board: np.ndarray, piece: int) -> int:
    frontier = 0
    positions = np.argwhere(board == piece)
    for x, y in positions:
        for dx, dy in NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and board[nx, ny] == 0:
                frontier += 1
                break
    return frontier


def evaluate(board: np.ndarray, player: int) -> float:
    empties = int(np.count_nonzero(board == 0))

    # Positional score
    # board has +1 for white and -1 for black
    raw_pos = float(np.sum(POS_WEIGHTS * board))
    pos_score = raw_pos if player == 1 else -raw_pos

    # Piece difference (more important late game)
    my_pieces = int(np.count_nonzero(board == player))
    opp_pieces = int(np.count_nonzero(board == -player))
    piece_diff = my_pieces - opp_pieces

    # Mobility (more important early/mid game)
    g_tmp = clone_game_from_board(board)
    my_moves = len(valid_moves(g_tmp, player))
    opp_moves = len(valid_moves(g_tmp, -player))
    mobility_diff = my_moves - opp_moves

    # Corner control (always important)
    my_corners = sum(1 for (x, y) in CORNERS if board[x, y] == player)
    opp_corners = sum(1 for (x, y) in CORNERS if board[x, y] == -player)
    corner_diff = my_corners - opp_corners

    # Frontier penalty (unstable pieces)
    my_frontier = count_frontier(board, player)
    opp_frontier = count_frontier(board, -player)
    # Fewer frontier pieces is better
    frontier_score = (opp_frontier - my_frontier)

    # Phase-based weights
    # Early: prioritize mobility/position/corners; downweight piece count
    # Late: prioritize piece count/corners
    if empties >= 44:  # early game
        w_pos, w_mob, w_corner, w_piece, w_frontier = 1.0, 6.0, 40.0, 0.5, 3.0
    elif empties >= 20:  # mid game
        w_pos, w_mob, w_corner, w_piece, w_frontier = 1.2, 5.0, 45.0, 1.5, 2.5
    else:  # late game
        w_pos, w_mob, w_corner, w_piece, w_frontier = 0.8, 2.0, 60.0, 8.0, 0.5

    score = (
        w_pos * pos_score
        + w_mob * mobility_diff
        + w_corner * corner_diff
        + w_piece * piece_diff
        + w_frontier * frontier_score
    )
    return float(score)


def ordered_moves(g: reversi, turn: int):
    moves = valid_moves(g, turn)

    def move_key(m):
        x, y = m
        # corners highest priority
        if (x, y) in CORNERS:
            return 10_000
        # otherwise use positional weight (from current player's perspective)
        return int(POS_WEIGHTS[x, y])

    # sort descending by key
    return sorted(moves, key=move_key, reverse=True)


def minimax(g: reversi, depth: int, alpha: float, beta: float, turn: int, player: int) -> float:
    board = g.board

    # Terminal / cutoff
    moves = valid_moves(g, turn)
    opp_moves = valid_moves(g, -turn)

    if depth == 0 or (not moves and not opp_moves):
        return evaluate(board, player)

    # If current side has no move, they pass
    if not moves:
        return minimax(g, depth - 1, alpha, beta, -turn, player)

    if turn == player:  # maximizing
        best = -float("inf")
        for (x, y) in ordered_moves(g, turn):
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)
            val = minimax(ng, depth - 1, alpha, beta, -turn, player)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:  # minimizing
        best = float("inf")
        for (x, y) in ordered_moves(g, turn):
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)
            val = minimax(ng, depth - 1, alpha, beta, -turn, player)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def choose_move(board: np.ndarray, player: int, depth: int = DEPTH):
    g = clone_game_from_board(board)
    moves = ordered_moves(g, player)

    if not moves:
        return (-1, -1)  # pass

    best_val = -float("inf")
    best_moves = []

    alpha, beta = -float("inf"), float("inf")

    for (x, y) in moves:
        ng = clone_game_from_board(board)
        ng.step(x, y, player, commit=True)
        val = minimax(ng, depth - 1, alpha, beta, -player, player)

        if val > best_val:
            best_val = val
            best_moves = [(x, y)]
        elif val == best_val:
            best_moves.append((x, y))

        alpha = max(alpha, best_val)

    return random.choice(best_moves)


def main():
    game_socket = socket.socket()
    game_socket.connect((HOST, PORT))

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        x, y = choose_move(board, turn, DEPTH)

        game_socket.send(pickle.dumps((x, y)))


if __name__ == "__main__":
    main()