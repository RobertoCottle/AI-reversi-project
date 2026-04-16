import time
import numpy as np
import socket
import pickle
from reversi import reversi
from core.encoder import get_legal_moves, apply_move


POSITIONAL_WEIGHTS = [
    100, -25,  10,  5,  5,  10, -25, 100,
    -25, -50,  -2, -2, -2,  -2, -50, -25,
     10,  -2,   5,  1,  1,   5,  -2,  10,
      5,  -2,   1,  2,  2,   1,  -2,   5,
      5,  -2,   1,  2,  2,   1,  -2,   5,
     10,  -2,   5,  1,  1,   5,  -2,  10,
    -25, -50,  -2, -2, -2,  -2, -50, -25,
    100, -25,  10,  5,  5,  10, -25, 100,
]


def positional_score(board: np.ndarray, player: int) -> float:
    score = 0.0
    for x in range(8):
        for y in range(8):
            w = POSITIONAL_WEIGHTS[x * 8 + y]
            if board[x, y] == player:
                score += w
            elif board[x, y] == -player:
                score -= w
    return score


def mobility_score(board: np.ndarray, player: int) -> float:
    my_moves  = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, -player))
    total = my_moves + opp_moves
    if total == 0:
        return 0.0
    return (my_moves - opp_moves) / total * 100.0


def corner_score(board: np.ndarray, player: int) -> float:
    corners = [(0,0),(0,7),(7,0),(7,7)]
    score   = 0.0
    for cx, cy in corners:
        if board[cx, cy] == player:
            score += 25.0
        elif board[cx, cy] == -player:
            score -= 25.0
    return score


def evaluate(board: np.ndarray, player: int) -> float:
    """
    Evaluate board from `player`s perspective.
    Positive = good for player, negative = bad for player.
    """
    total_pieces = int(np.sum(board != 0))

    pos  = positional_score(board, player)
    mob  = mobility_score(board, player)
    corn = corner_score(board, player)

    if total_pieces > 50:
        my  = int(np.sum(board == player))
        opp = int(np.sum(board == -player))
        disc = float(my - opp) * 10.0
    else:
        disc = 0.0

    return pos + mob + corn + disc


def order_moves(legal: list) -> list:
    corners  = {(0,0),(0,7),(7,0),(7,7)}
    xsquares = {(1,1),(1,6),(6,1),(6,6)}
    priority, middle, last = [], [], []
    for move in legal:
        if move in corners:
            priority.append(move)
        elif move in xsquares:
            last.append(move)
        else:
            middle.append(move)
    return priority + middle + last


def negamax(
    board:      np.ndarray,
    player:     int,
    depth:      int,
    alpha:      float,
    beta:       float,
    start_time: float = None,
    time_limit: float = None,
) -> float:
    """
    Negamax — a cleaner minimax formulation.

    Always maximizes from the current player's perspective.
    At each level the score is negated when returning to parent,
    so the parent automatically gets the opponent's score negated.

    This eliminates the maximizing/minimizing flag bug entirely.
    """
    # Time check
    if start_time is not None and time_limit is not None:
        if time.time() - start_time > time_limit * 0.9:
            return evaluate(board, player)

    legal = get_legal_moves(board, player)

    # Terminal or leaf
    if depth == 0:
        return evaluate(board, player)

    if not legal:
        # Check if opponent has moves
        opp_legal = get_legal_moves(board, -player)
        if not opp_legal:
            # Game over — return disc count
            my  = int(np.sum(board == player))
            opp = int(np.sum(board == -player))
            if my > opp:
                return 10000.0
            elif opp > my:
                return -10000.0
            return 0.0
        # Current player passes
        return -negamax(board, -player, depth - 1,
                       -beta, -alpha, start_time, time_limit)

    ordered = order_moves(legal)
    value   = -np.inf

    for (x, y) in ordered:
        if start_time is not None and time_limit is not None:
            if time.time() - start_time > time_limit * 0.9:
                break

        child = apply_move(board, x, y, player)
        score = -negamax(child, -player, depth - 1,
                        -beta, -alpha, start_time, time_limit)
        value = max(value, score)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return value


def select_move(board: np.ndarray, player: int, depth: int = 5) -> tuple:
    """Select best move using negamax without time limit."""
    legal = get_legal_moves(board, player)
    if not legal:
        return -1, -1

    best_move  = legal[0]
    best_score = -np.inf
    ordered    = order_moves(legal)

    for (x, y) in ordered:
        child = apply_move(board, x, y, player)
        score = -negamax(child, -player, depth - 1,
                        -np.inf, np.inf)
        if score > best_score:
            best_score = score
            best_move  = (x, y)

    return best_move


def select_move_timed(
    board:      np.ndarray,
    player:     int,
    time_limit: float = 5.0,
    max_depth:  int   = 5,
) -> tuple:
    """
    Iterative deepening negamax with time limit.
    Searches depth 1, 2, 3... up to max_depth.
    Returns best move found within time limit.
    """
    legal = get_legal_moves(board, player)
    if not legal:
        return -1, -1

    best_move   = order_moves(legal)[0]
    start_time  = time.time()

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit * 0.9:
            break

        depth_best_move  = best_move
        depth_best_score = -np.inf
        ordered          = order_moves(legal)

        for (x, y) in ordered:
            if time.time() - start_time > time_limit * 0.9:
                break

            child = apply_move(board, x, y, player)
            score = -negamax(child, -player, depth - 1,
                            -np.inf, np.inf, start_time, time_limit)

            if score > depth_best_score:
                depth_best_score = score
                depth_best_move  = (x, y)

        if time.time() - start_time < time_limit * 0.9:
            best_move = depth_best_move

    return best_move


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        x, y = select_move_timed(board, turn, time_limit=5.0, max_depth=5)
        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()