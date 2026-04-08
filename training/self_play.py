import os
import copy
import numpy as np
import torch

from reversi_ai.reversi import reversi as reversi_game
from bootstrap.td_selfplay import get_legal_moves
from model.encoder import encode_board
from model.network import PolicyValueNet
from search.mcts import MonteCarloTreeSearch
from training.replay_store import ReplayStore


class SelfPlayWorker:
    #Plays complete games of Reversi using MCTS and records every
    #position as a training example.

    #The temperature schedule controls exploration:
    #    - First `temp_threshold` moves: temperature=1.0 (exploratory)
    #    - After that: temperature→0 (near-deterministic, best move)

    #This means opening play is varied across games (good for diversity
    #in the replay buffer) while endgame play is precise.
    

    def __init__(
        self,
        model:           PolicyValueNet,
        device:          torch.device,
        n_simulations:   int   = 400,
        c_puct:          float = 1.5,
        temp_threshold:  int   = 12,
        add_noise:       bool  = True,
    ):
        self.model          = model
        self.device         = device
        self.n_simulations  = n_simulations
        self.c_puct         = c_puct
        self.temp_threshold = temp_threshold
        self.add_noise      = add_noise
        self.mcts           = MonteCarloTreeSearch(model, device, c_puct)

    def play_one_game(self) -> list:
        #Play one complete self-play game from the opening position.

        #Returns:
        #    trajectory: list of (board_tensor, policy, value) triples
        #                value is filled in retrospectively after game ends
        
        game      = reversi_game()
        history   = []   # (board_tensor, policy, player)
        move_count = 0
        consecutive_passes = 0

        while True:
            player = game.turn
            legal  = get_legal_moves(game, player)

            if not legal:
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
                # Pass — record as (-1,-1) and flip turn
                history.append((
                    encode_board(game.board, player),
                    self._pass_policy(),
                    player,
                ))
                game.turn = -game.turn
                move_count += 1
                continue
            else:
                consecutive_passes = 0

            # Temperature schedule
            temperature = 1.0 if move_count < self.temp_threshold else 0.0

            # Run MCTS search
            policy, _ = self.mcts.search(
                board         = game.board,
                player        = player,
                n_simulations = self.n_simulations,
                add_noise     = self.add_noise,
            )

            # Record position before move
            history.append((
                encode_board(game.board, player),
                policy.copy(),
                player,
            ))

            # Select and apply action
            action = self.mcts.select_action(policy, temperature=temperature)
            x, y   = action

            if x == -1 and y == -1:
                game.turn = -game.turn
            else:
                game.step(x, y, player, commit=True)
                game.turn = -game.turn

            move_count += 1

        # Determine game outcome
        white = int(np.sum(game.board == 1))
        black = int(np.sum(game.board == -1))

        if white > black:
            outcome = 1.0
        elif black > white:
            outcome = -1.0
        else:
            outcome = 0.0

        # Label each position with outcome from that player's perspective
        trajectory = []
        for board_tensor, policy, player in history:
            value = outcome if player == 1 else -outcome
            trajectory.append((board_tensor, policy, value))

        return trajectory, outcome

    def _pass_policy(self) -> np.ndarray:
        # Return a policy vector with all probability on the pass action.
        policy     = np.zeros(65, dtype=np.float32)
        policy[64] = 1.0
        return policy


def run_self_play(
    checkpoint_path: str  = "data/checkpoints/pretrained.pt",
    buffer_path:     str  = "data/replay_buffer/buffer.npz",
    n_games:         int  = 200,
    n_simulations:   int  = 400,
    c_puct:          float = 1.5,
    temp_threshold:  int  = 12,
    log_every:       int  = 10,
):
    #Run n_games of self-play and save all positions to the replay buffer.

    #This is called once before training begins to populate the buffer,
    #and then called again periodically during the training loop to keep
    #the buffer fresh with positions from the latest network version.
    
    os.makedirs("data/replay_buffer", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SelfPlay] Using device: {device}")

    # Load network
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PolicyValueNet(
        channels = checkpoint['channels'],
        n_blocks = checkpoint['n_blocks'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[SelfPlay] Loaded network from {checkpoint_path}")

    # Load or create replay buffer
    store = ReplayStore(max_size=500_000, save_path=buffer_path)
    store.load()

    worker = SelfPlayWorker(
        model          = model,
        device         = device,
        n_simulations  = n_simulations,
        c_puct         = c_puct,
        temp_threshold = temp_threshold,
        add_noise      = True,
    )

    white_wins = 0
    black_wins = 0
    draws      = 0

    print(f"[SelfPlay] Playing {n_games} games...")
    for game_idx in range(n_games):
        trajectory, outcome = worker.play_one_game()
        store.push_game(trajectory)

        if outcome > 0:
            white_wins += 1
        elif outcome < 0:
            black_wins += 1
        else:
            draws += 1

        if (game_idx + 1) % log_every == 0:
            total = game_idx + 1
            print(
                f"  Game {total:>4}/{n_games} | "
                f"Buffer: {len(store):,} | "
                f"W={white_wins/total:.2%} "
                f"B={black_wins/total:.2%} "
                f"D={draws/total:.2%}"
            )

    store.save()
    print(f"[SelfPlay] Done. Buffer saved to {buffer_path}")
    print(f"[SelfPlay] Total positions in buffer: {len(store):,}")