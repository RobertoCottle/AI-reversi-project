# I've done a lot of finnangling, so far, this is the best one, but still room for improvement
# Hybrid Evaluation + Minimax (alpha-beta) + Transposition Table + Dynamic Depth

import socket
import pickle
import numpy as np
from reversi import reversi

HOST = "127.0.0.1"
PORT = 33333

# Base search depth (increases in late-game)
BASE_DEPTH = 3

# Positional Strategy Board - weighted positioning
# Can modify / I've been playing with it
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

# Transposition Table
# TT eliminates duplicate work within the same move calculation.
# key: (board_bytes, turn, depth, root_player) -> minimax value
TT = {}


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8


def clone_game_from_board(board: np.ndarray) -> reversi:
    g = reversi()
    g.board = board.copy()
    # Keep counts consistent
    g.white_count = int(np.count_nonzero(g.board == 1))
    g.black_count = int(np.count_nonzero(g.board == -1))
    return g

#all legal moves for turn
def valid_moves(g: reversi, turn: int):
    moves = []
    for i in range(8):
        for j in range(8):
            flips = g.step(i, j, turn, commit=False)
            if flips > 0:
                moves.append((i, j))
    return moves

# Frontier pieces are adjacent to at least one empty square
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
    empties = int(np.count_nonzero(board == 0)) #checks stategy change by phase, early, mid, late Chnage weights based on empties

    # Positional score
    # board has +1 for white and -1 for black
    raw_pos = float(np.sum(POS_WEIGHTS * board))
    pos_score = raw_pos if player == 1 else -raw_pos

    # Piece difference (more important late game), early disc adv is bad, late disc is wins the game
    my_pieces = int(np.count_nonzero(board == player))
    opp_pieces = int(np.count_nonzero(board == -player))
    piece_diff = my_pieces - opp_pieces

    # Mobility (more important early/mid game), number of legal moves, controls tempo, restrict their options
    g_tmp = clone_game_from_board(board)
    my_moves = len(valid_moves(g_tmp, player))
    opp_moves = len(valid_moves(g_tmp, -player))
    mobility_diff = my_moves - opp_moves

    # Corner control (always important)
    my_corners = sum(1 for (x, y) in CORNERS if board[x, y] == player)
    opp_corners = sum(1 for (x, y) in CORNERS if board[x, y] == -player)
    corner_diff = my_corners - opp_corners #corners often decide game

    # Frontier penalty (unstable pieces)
    my_frontier = count_frontier(board, player)
    opp_frontier = count_frontier(board, -player)
    # Fewer frontier pieces is better
    frontier_score = (opp_frontier - my_frontier) #frontier pieces are vulnerable, give opponent flipping opportunities, increase opponent mobilitiy, reward fewer frontier piecies

    # Phase-based weights
    # Early: prioritize mobility/position/corners; downweight piece count
    # Late: prioritize piece count/corners
    # Should tinker with these values - noticed corner hesitation mid-game
    if empties >= 44:  # early
        w_pos, w_mob, w_corner, w_piece, w_frontier = 1.0, 6.0, 40.0, 0.5, 3.0
    elif empties >= 20:  # mid
        w_pos, w_mob, w_corner, w_piece, w_frontier = 1.2, 5.0, 45.0, 1.5, 2.5
    else:  # late
        w_pos, w_mob, w_corner, w_piece, w_frontier = 0.8, 2.0, 60.0, 8.0, 0.5

    return float(
        w_pos * pos_score
        + w_mob * mobility_diff
        + w_corner * corner_diff
        + w_piece * piece_diff
        + w_frontier * frontier_score
    )

# Ordering: Corners first, then positional weight, sorts by strategic priority before searching them
#searching first makes pruning stronger, with ordering alpha-beter approaches optimal pruning
def ordered_moves(g: reversi, turn: int):
    moves = valid_moves(g, turn)

    def move_key(m):
        x, y = m
        if (x, y) in CORNERS: #corners are huge priority, make edges stable, increase win in probability, most valuable scaures on board
            return 10_000
        return int(POS_WEIGHTS[x, y]) # POS

    return sorted(moves, key=move_key, reverse=True)

# Dynamic Depth - should tinker with these values as well as monitor results
def choose_depth(board: np.ndarray) -> int:
    empties = int(np.count_nonzero(board == 0))
    d = BASE_DEPTH
    if empties <= 14:
        d += 1
    if empties <= 8:
        d += 1
    return d


def tt_key(board: np.ndarray, turn: int, depth: int, root_player: int):
    return (board.tobytes(), turn, depth, root_player)

# minimax + alpha-beta, plus transposition-table
def minimax(g: reversi, depth: int, alpha: float, beta: float, turn: int, root_player: int) -> float:
    board = g.board
    k = tt_key(board, turn, depth, root_player)
    if k in TT:
        return TT[k]

    moves = valid_moves(g, turn)
    opp_moves = valid_moves(g, -turn)

    # Terminal / cutoff
    if depth == 0 or (not moves and not opp_moves):
        val = evaluate(board, root_player)
        TT[k] = val
        return val

    # Pass handling
    if not moves:
        val = minimax(g, depth - 1, alpha, beta, -turn, root_player)
        TT[k] = val
        return val

    if turn == root_player:  # maximizing
        best = -float("inf")
        for (x, y) in ordered_moves(g, turn):
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)
            val = minimax(ng, depth - 1, alpha, beta, -turn, root_player)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if beta <= alpha:
                break
        TT[k] = best
        return best
    else:  # minimizing
        best = float("inf")
        for (x, y) in ordered_moves(g, turn):
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)
            val = minimax(ng, depth - 1, alpha, beta, -turn, root_player)
            if val < best:
                best = val
            if best < beta:
                beta = best
            if beta <= alpha:
                break
        TT[k] = best
        return best


def choose_move(board: np.ndarray, player: int):
    g = clone_game_from_board(board)
    moves = ordered_moves(g, player)
    if not moves:
        return (-1, -1)

    depth = choose_depth(board)

    best_move = moves[0]
    best_val = -float("inf")
    alpha, beta = -float("inf"), float("inf")

    # tie behavior chooses "first" found (avoids randomization). Ideally, we should handle ties with strategy
    # i.e. minimizing frontier, proximity to other advantageous positions, etc.
    for (x, y) in moves:
        ng = clone_game_from_board(board)
        ng.step(x, y, player, commit=True)
        val = minimax(ng, depth - 1, alpha, beta, -player, player)
        if val > best_val:
            best_val = val
            best_move = (x, y)
        if best_val > alpha:
            alpha = best_val

    return best_move


def main():
    game_socket = socket.socket()
    game_socket.connect((HOST, PORT))

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        x, y = choose_move(board, turn)
        game_socket.send(pickle.dumps((x, y)))


if __name__ == "__main__":
    main()