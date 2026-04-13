import numpy as np
import copy
from reversi_ai.reversi import reversi as reversi_game
from bootstrap.ntuple_networks import NTupleNetwork

def get_legal_moves_from_board(board: np.ndarray, player: int) -> list:
    """
    Get legal moves directly from a board array without using
    a reversi game instance, avoiding any mutation risk.
    """
    directions = [
        [1,1],[1,0],[1,-1],[0,1],
        [0,-1],[-1,1],[-1,0],[-1,-1]
    ]
    legal = []
    for x in range(8):
        for y in range(8):
            if board[x, y] != 0:
                continue
            for dx, dy in directions:
                cx, cy = x + dx, y + dy
                found_opponent = False
                while 0 <= cx <= 7 and 0 <= cy <= 7:
                    if board[cx, cy] == 0:
                        break
                    elif board[cx, cy] == -player:
                        found_opponent = True
                        cx += dx
                        cy += dy
                    elif board[cx, cy] == player:
                        if found_opponent:
                            legal.append((x, y))
                        break
                    else:
                        break
                if (x, y) in legal:
                    break
    return list(dict.fromkeys(legal))  # deduplicate preserving order

def get_legal_moves(game: reversi_game, player: int) -> list[tuple[int, int]]:
    
    #Probe all 64 squares using commit=False to find legal moves
    #for `player` without mutating the game state.

    #Returns list of (x, y) tuples. Empty list means the player must pass.
    
    legal = []
    for x in range(8):
        for y in range(8):
            if game.step(x, y, player, commit=False) > 0:
                legal.append((x, y))
    return legal


def play_one_game(
    net: NTupleNetwork,
    epsilon: float = 0.1,
) -> list[tuple[np.ndarray, int, float]]:
    
    #Play one complete self-play game using the n-tuple network as a
    #1-ply greedy value function, with epsilon-greedy exploration.

    #At each turn:
    #  1. Enumerate legal moves for the current player.
    #  2. With probability epsilon, pick a random legal move.
    #  3. Otherwise, pick the move that maximises net.evaluate_for_player()
    #     on the resulting board (1-ply lookahead, commit=False + deepcopy).
    #  4. Apply the chosen move.

    #Returns:
    #    List of (board_snapshot, player, placeholder_value) triples,
    #    one per move. The placeholder_value is filled in by the caller
    #    after the game ends (retrospective TD target assignment).
    
    game = reversi_game()
    history = []   # (board_copy, player)
    consecutive_passes = 0

    while True:
        player = game.turn
        legal = get_legal_moves(game, player)

        if not legal:
            # No legal moves: player must pass
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break   # Both players passed — game over
            game.turn = -game.turn
            continue
        else:
            consecutive_passes = 0

        # Choose move: epsilon-greedy 1-ply lookahead
        if np.random.random() < epsilon:
            x, y = legal[np.random.randint(len(legal))]
        else:
            best_val = -np.inf
            best_move = legal[0]
            for (x, y) in legal:
                # Simulate move on a deep copy to not mutate game state
                sim = copy.deepcopy(game)
                sim.step(x, y, player, commit=True)
                val = net.evaluate_for_player(sim.board, player)
                if val > best_val:
                    best_val = val
                    best_move = (x, y)
            x, y = best_move

        # Record state BEFORE move (what position led to this decision)
        history.append((game.board.copy(), player))

        # Apply move
        game.step(x, y, player, commit=True)
        game.turn = -game.turn

    # Determine outcome from white's perspective
    if game.white_count > game.black_count:
        outcome = 1.0    # white wins
    elif game.black_count > game.white_count:
        outcome = -1.0   # black wins
    else:
        outcome = 0.0    # draw

    # Attach TD targets: final outcome for every position
    # (simple Monte Carlo return — no bootstrapping yet)
    labelled = [
        (board, player, outcome if player == 1 else -outcome)
        for (board, player) in history
    ]
    return labelled