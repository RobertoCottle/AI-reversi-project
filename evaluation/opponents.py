import numpy as np
import copy
from reversi_ai.reversi import reversi as reversi_game
from bootstrap.td_selfplay import get_legal_moves


class RandomAgent:
    # Picks a random legal move every turn.
    # Baseline — a trained agent should beat this 99%+ of the time.
    
    def select_action(self, board: np.ndarray, player: int):
        game = reversi_game()
        game.board = board.copy()
        legal = get_legal_moves(game, player)
        if not legal:
            return (-1, -1)
        return legal[np.random.randint(len(legal))]


class GreedyAgent:
    
    #Always picks the move that flips the most discs immediately.
    #Understands rules but has no strategic awareness.
    #A trained agent should beat this consistently.
    
    def select_action(self, board: np.ndarray, player: int):
        game = reversi_game()
        game.board = board.copy()
        legal = get_legal_moves(game, player)
        if not legal:
            return (-1, -1)

        best_move   = legal[0]
        best_flips  = -1

        for (x, y) in legal:
            flips = game.step(x, y, player, commit=False)
            if flips > best_flips:
                best_flips = flips
                best_move  = (x, y)
            # restore board after probe — commit=False does not mutate
        return best_move


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
    
    #Score a board position using the static positional weight table.
    #Positive = good for player, negative = bad for player.
    
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

    #Mobility: difference in number of legal moves available.
    #More moves = more options = strategic advantage.
    
    game_p = reversi_game()
    game_p.board = board.copy()
    my_moves  = len(get_legal_moves(game_p, player))

    game_o = reversi_game()
    game_o.board = board.copy()
    opp_moves = len(get_legal_moves(game_o, -player))

    if my_moves + opp_moves == 0:
        return 0.0
    return (my_moves - opp_moves) / (my_moves + opp_moves)


class AlphaBetaAgent:

    #Minimax search with alpha-beta pruning and a hand-crafted
    #evaluation function combining positional weights and mobility.

    #depth=1 is weak but fast.
    #depth=3 is moderate — a real strategic challenge for early training.
    #depth=5 is strong — beating this consistently means the agent has
    #        developed genuine positional understanding.
    

    def __init__(self, depth: int = 3):
        self.depth = depth

    def select_action(self, board: np.ndarray, player: int):
        game = reversi_game()
        game.board = board.copy()
        legal = get_legal_moves(game, player)
        if not legal:
            return (-1, -1)

        best_move  = legal[0]
        best_score = -np.inf

        for (x, y) in legal:
            child = reversi_game()
            child.board = board.copy()
            child.step(x, y, player, commit=True)
            score = self._minimax(
                child.board, -player, self.depth - 1,
                -np.inf, np.inf, False, player
            )
            if score > best_score:
                best_score = score
                best_move  = (x, y)

        return best_move

    def _evaluate(self, board: np.ndarray, player: int) -> float:
        pos = positional_score(board, player)
        mob = mobility_score(board, player) * 50.0
        return pos + mob

    def _minimax(
        self,
        board:      np.ndarray,
        current:    int,
        depth:      int,
        alpha:      float,
        beta:       float,
        maximizing: bool,
        root_player: int,
    ) -> float:
        game = reversi_game()
        game.board = board.copy()
        legal = get_legal_moves(game, current)

        if depth == 0 or not legal:
            return self._evaluate(board, root_player)

        if maximizing:
            value = -np.inf
            for (x, y) in legal:
                child = reversi_game()
                child.board = board.copy()
                child.step(x, y, current, commit=True)
                value = max(value, self._minimax(
                    child.board, -current, depth - 1,
                    alpha, beta, False, root_player
                ))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for (x, y) in legal:
                child = reversi_game()
                child.board = board.copy()
                child.step(x, y, current, commit=True)
                value = min(value, self._minimax(
                    child.board, -current, depth - 1,
                    alpha, beta, True, root_player
                ))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value