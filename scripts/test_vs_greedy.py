import numpy as np
import torch
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves, encode_board
from network.q_network import QNetwork


MODEL_PATH = "saved/models/pretrained_q.pt"


def greedy_move(board: np.ndarray, player: int):
    """Pick the move that flips the most discs."""
    legal = get_legal_moves(board, player)
    if not legal:
        return -1, -1

    best_flips = -1
    best_x, best_y = legal[0]

    for lx, ly in legal:
        sim = reversi_game()
        sim.board = board.copy()
        flips = sim.step(lx, ly, player, commit=False)
        if flips > best_flips:
            best_flips = flips
            best_x, best_y = lx, ly

    return best_x, best_y


def play_game(model, device, neural_is_white: bool, debug: bool = False) -> int:
    game = reversi_game()
    consecutive_passes = 0
    move_count = 0

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

        is_neural = (player == 1 and neural_is_white) or \
                    (player == -1 and not neural_is_white)

        if is_neural:
            x, y = model.predict_move(game.board, player, legal, device)
        else:
            x, y = greedy_move(game.board, player)

        if x == -1 and y == -1:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.turn = -game.turn
            continue

        consecutive_passes = 0
        result = game.step(x, y, player, commit=True)

        if result < 0 and legal:
            x, y = legal[0]
            game.step(x, y, player, commit=True)

        game.turn = -game.turn
        move_count += 1

        if debug:
            w = int(np.sum(game.board == 1))
            b = int(np.sum(game.board == -1))
            agent = "Neural" if is_neural else "Greedy"
            print(f"  Move {move_count:>2}: {agent} "
                  f"({'W' if player==1 else 'B'}) "
                  f"({x},{y}) | W={w} B={b}")

    white = int(np.sum(game.board == 1))
    black = int(np.sum(game.board == -1))

    if debug:
        print(f"  Final: W={white} B={black}")

    if white > black:
        return 1
    elif black > white:
        return -1
    return 0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Try final model first, fall back to pretrained
    import os
    model_path = "saved/models/final_q.pt" \
                 if os.path.exists("saved/models/final_q.pt") \
                 else MODEL_PATH

    model = QNetwork.load(model_path, device)
    model.eval()

    n_games = 10

    print(f"\nNeural Q-network vs Greedy ({n_games} games each)")
    print("=" * 50)

    print(f"\nNeural as WHITE:")
    wins = losses = draws = 0
    for i in range(n_games):
        outcome = play_game(model, device, neural_is_white=True)
        if outcome == 1:
            wins += 1
            result = "neural wins"
        elif outcome == -1:
            losses += 1
            result = "greedy wins"
        else:
            draws += 1
            result = "draw"
        print(f"  Game {i+1:>2}: {result}")
    print(f"  Win rate: {wins}/{n_games} "
          f"({wins/n_games:.0%} W, {losses/n_games:.0%} L, "
          f"{draws/n_games:.0%} D)")

    print(f"\nNeural as BLACK:")
    wins = losses = draws = 0
    for i in range(n_games):
        outcome = play_game(model, device, neural_is_white=False)
        if outcome == -1:
            wins += 1
            result = "neural wins"
        elif outcome == 1:
            losses += 1
            result = "greedy wins"
        else:
            draws += 1
            result = "draw"
        print(f"  Game {i+1:>2}: {result}")
    print(f"  Win rate: {wins}/{n_games} "
          f"({wins/n_games:.0%} W, {losses/n_games:.0%} L, "
          f"{draws/n_games:.0%} D)")