# I've done a lot of finnangling, so far, this is the best one, but still room for improvement
# Hybrid Evaluation + Minimax (alpha-beta) + Transposition Table + Dynamic Depth

import socket
import pickle
import numpy as np
from reversi import reversi
import random

HOST = "127.0.0.1"
PORT = 33333

#At BASE_DEPTH 6, avg win count increase 2x compared to Better_Player.py, at depth 7, BPV3 is able to beat Better_player
#but it has an avg decision time of 15 seconds, sometimes going even longer than a minute. 

# Base search depth (increases in late-game)
BASE_DEPTH = 6 # depth 6, lead to an average of a 6 second decision, 9 exceeded past a minute --- 3/3/26

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

# Zobristhashing is what is allowing the TT table to be used at a greater depth
#XOR allows us to recompute only the cells that changed instead of rehasing all 64 squares from scratch
#Instead of storing the entire board (64 cells) as the transposition table key, 
# you store a single integer. Comparing two integers is vastly cheaper than comparing two arrays.
class ZobristHasher:
    def __init__(self):
        # 64 cells, 3 states: 0=Empty, 1=White, -1=black
        # We use a dictonary or list to mape these states to indicies
        self.state_map = {0: 0, 1: 1, -1: 2}
        # Table: [64_cells][3_states]
        self.table = [[random.getrandbits(64) for _ in range(3)] for _ in range(64)]

    def get_hash(self, board: np.ndarray) -> int:
        h = 0
        for i in range(8):
            for j in range(8):
                piece = board[i, j]
                h ^= self.table[i * 8 + j][self.state_map[piece]]
        return h
    
    def update_hash(self, current_hash: int, x: int, y: int, old_piece: int, new_piece: int) -> int:
        # Remove the old piece state
        current_hash ^= self.table[x * 8 + y][self.state_map[old_piece]]
        # Add the new piece state
        current_hash ^= self.table[x * 8 + y][self.state_map[new_piece]]
        return current_hash
    
# Initialize globally
hasher = ZobristHasher()

def make_child_hash(old_board: np.ndarray, new_board: np.ndarray, current_hash: int) -> int:
        # Compute new Zobrist hash by XOR-ing only the cells that changed."""
        h = current_hash
        for i in range(8):
            for j in range(8):
                if old_board[i, j] != new_board[i, j]:
                    h = hasher.update_hash(h, i, j, old_board[i, j], new_board[i, j])
        return h


### function for detecting wedges = A "wedge" move occurs when you play into an empty square that 
### is sandwiched between two opponent stones on an edge. This prevents the opponent from "flipping" that whole edge segment back.

# Count pieces that cannot be flipped,
# corners are stable, edges are stable
def count_stable_discs(board: np.ndarray, player: int) -> int: 
    stable_count = 0
    # A full stability check is computationally expensive.
    # We'll use a simplified weighted sum for the corners + adjacent edges.
    for x, y in CORNERS:
        if board[x, y] == player:
            stable_count += 5 #this should not be a static number, needs to be updated to dynamic, once earlier version is completed
    return stable_count

#returns 1 if we have parity control (forcing opponent to move) otherwise 0
def get_parity_score(board: np.ndarray, player: int) -> int: #player depends on whose turn it is
    empties = int(np.count_nonzero(board == 0))
    # If empties is even, the current player has the parity advantage.
    return 1 if empties % 2 == 0 else 0


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


### Parity and Stability is needed in the evaluate function for the evaporation strategy
### In late game, you want to maintain even number of squares if the second player, 
### or odd numbers if you are the first.
### Stability: This is the antidote to "flippability." A disc is "stable" if it cannot be 
### flipped back for the remainder of the game (e.g., corners and edges protected by other stable discs).
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

    #for evaporation control
    stability_score = count_stable_discs(board, player) - count_stable_discs(board, -player)
    parity_advantage = get_parity_score(board, player)
    
    # Phase-based weights
    # Early: prioritize mobility/position/corners; downweight piece count
    # Late: prioritize piece count/corners
    # Should tinker with these values - noticed corner hesitation mid-game
    if empties >= 44:  # early
        w_pos, w_mob, w_corner, w_piece, w_frontier, w_parity, w_stability  = 1.0, 6.0, 40.0, 0.5, 3.0, 10.0, 2.0
    elif empties >= 20:  # mid
        w_pos, w_mob, w_corner, w_piece, w_frontier, w_parity, w_stability = 1.2, 5.0, 45.0, 1.5, 2.5, 5.0, 5.0
    else:  # late
        w_pos, w_mob, w_corner, w_piece, w_frontier, w_parity, w_stability = 0.8, 2.0, 60.0, 8.0, 0.5, 2.0, 15.0

    return float(
        w_pos * pos_score
        + w_mob * mobility_diff
        + w_corner * corner_diff
        + w_piece * piece_diff
        + w_frontier * frontier_score
        + w_parity * parity_advantage
        + w_stability * stability_score
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
def minimax(g: reversi, depth: int, alpha: float, beta: float, turn: int, root_player: int, board_hash: int) -> float:
    board = g.board
    # board.tobytes() was converting 8x8 numpy array into a
    # 64-byte string every single time a node is visited.
    #with hash, you only XOR the cells have changed, not the whole board
    k = (board_hash, turn, depth, root_player)
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
        val = minimax(g, depth - 1, alpha, beta, -turn, root_player, board_hash)
        TT[k] = val
        return val

    if turn == root_player:  # maximizing
        best = -float("inf")
        for (x, y) in ordered_moves(g, turn):
            old_board = g.board.copy()
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)

            # Calculate new hash incrementally (pseduocode)
            # You need to track which cells changed during ng.step()
            # and call hasher.update_hash(...) for each one
            ## new_hash = .... # updated via hasher.update_hash
            #  def get_hash(self, board: np.ndarray) -> int:
            #  update_hash(self, current_hash: int, x: int, y: int, old_piece: int, new_piece: int) -> int: 
            #new_hash = hasher.update_hash(ng.board, x, y,  )
            new_hash = make_child_hash(old_board, ng.board, board_hash)
            # Calculate new
            val = minimax(ng, depth - 1, alpha, beta, -turn, root_player, new_hash)
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
            old_board = g.board.copy()
            ng = clone_game_from_board(board)
            ng.step(x, y, turn, commit=True)

            new_hash = make_child_hash(old_board, ng.board, board_hash)
            # Calculate new hash incrementally (pseduocode)
            # You need to track which cells changed during ng.step()
            # and call hasher.update_hash(...) for each one
            ## new_hash = .... # updated via hasher.update_hash

            val = minimax(ng, depth - 1, alpha, beta, -turn, root_player, new_hash)
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
        val = minimax(
            ng, 
            depth - 1, 
            alpha, 
            beta, 
            -player, 
            player, 
            hasher.get_hash(ng.board)
            )
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