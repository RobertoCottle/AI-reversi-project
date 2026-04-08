import os
import numpy as np
from bootstrap.ntuple_networks import NTupleNetwork
from bootstrap.td_selfplay import play_one_game

def run_bootstrap(
        n_games: int = 50_000,
        lr: float = 0.001,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        save_path: str = "data/checkpoints/ntuple_weights.pkl",
        log_every: int = 500,
):
    #phase 1 training loop.

    #Plays n_games self-play games, applies TD updates after each game, 
    #and decays epsilon linearly from epsilon_start to epsilon_end.

    #The learning rate is intentionally small = the n-tuple network
    #converges stably without any scheduling because the update rule
    #is bounded by the tanh gradient term (max delta per step = lr * 0.25).

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tuples = NTupleNetwork.make_standard_tuples()
    net = NTupleNetwork(tuples)

    print(f"[Bootstrap] Starting TD training: {n_games} games")
    print(f"[Bootstrap] Tuple count: {net.n_tuples}, weights per tuple: {net.n_patterns}")
    print(f"[Bootstrap] Total parameters: {net.n_tuples * net.n_patterns:,}")

    win_counts = {'white': 0, 'black': 0, 'draw': 0}

    for game_idx in range(n_games):

        # Linear epsilon decay
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (game_idx / n_games)

        # Play game and get labelled trajectory
        trajectory = play_one_game(net, epsilon)

        # TD update: apply updates in reverse order (from terminal backward)
        # This propagates outcome signal back through the game more efficently
        for board, player, target in reversed(trajectory):
            net.update(board, target if player == 1 else -target, lr=lr)

        # Track outcomes for logging
        _, _, final_target = trajectory[-1]
        if final_target > 0:
            win_counts['white'] += 1
        elif final_target < 0:
            win_counts['black'] += 1
        else:
            win_counts['draw'] += 1

        # Periodic logging
        if (game_idx + 1) % log_every == 0:
            total = game_idx + 1
            print(
                f"  Game {total:>6} / {n_games} | "
                f"ε={epsilon:.3f} | "
                f"W={win_counts['white']/total:.2%} "
                f"B={win_counts['black']/total:.2%} "
                f"D={win_counts['draw']/total:.2%}"
            )

    net.save(save_path)
    print(f"[Bootstrap] Training complete. Final network saved to {save_path}")
    return net

if __name__ == "__main__":
    run_bootstrap()