import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from reversi_ai.reversi import reversi as reversi_game
from bootstrap.ntuple_networks import NTupleNetwork
from bootstrap.td_selfplay import get_legal_moves, play_one_game
from model.encoder import encode_board
from model.network import PolicyValueNet


def generate_pretrain_dataset(
    ntuple_path: str,
    n_games: int = 5_000,
    device: torch.device = torch.device('cpu'),
):

    #Play n_games using the trained n-tuple network and record every
    #position alongside:
    #    - the n-tuple value estimate as the value target
    #    - a uniform distribution over legal moves as the policy target
    #      (we have no MCTS policy yet — uniform is a neutral prior)

    #Returns:
    #    board_tensors: torch.Tensor (N, 6, 8, 8)
    #    policy_targets: torch.Tensor (N, 65)
    #    value_targets: torch.Tensor (N, 1)
    
    print(f"[Pretrain] Loading n-tuple weights from {ntuple_path}")
    net = NTupleNetwork.load(ntuple_path)

    board_tensors  = []
    policy_targets = []
    value_targets  = []

    print(f"[Pretrain] Generating {n_games} games for pretraining dataset...")
    for game_idx in range(n_games):
        if (game_idx + 1) % 500 == 0:
            print(f"  Game {game_idx + 1} / {n_games}")

        trajectory = play_one_game(net, epsilon=0.1)

        for board, player, outcome in trajectory:
            # Encode board as tensor
            tensor = encode_board(board, player)
            board_tensors.append(tensor)

            # Value target: n-tuple estimate, from current player's perspective
            ntuple_val = net.evaluate_for_player(board, player)
            # Blend 50/50 with actual outcome for a more grounded target
            blended = 0.5 * ntuple_val + 0.5 * outcome
            value_targets.append([blended])

            # Policy target: uniform over legal moves
            # (MCTS in Layer 4 will produce real policy targets later)
            game = reversi_game()
            game.board = board.copy()
            legal = get_legal_moves(game, player)
            policy = np.zeros(65, dtype=np.float32)
            if legal:
                for (x, y) in legal:
                    policy[x * 8 + y] = 1.0
                policy /= policy.sum()
            else:
                policy[64] = 1.0   # pass action
            policy_targets.append(policy)

    board_tensors  = torch.stack(board_tensors)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32)
    value_targets  = torch.tensor(np.array(value_targets),  dtype=torch.float32)

    print(f"[Pretrain] Dataset size: {len(board_tensors):,} positions")
    return board_tensors, policy_targets, value_targets


def run_pretrain(
    ntuple_path:  str = "data/checkpoints/ntuple_weights.pkl",
    save_path:    str = "data/checkpoints/pretrained.pt",
    n_games:      int = 5_000,
    epochs:       int = 20,
    batch_size:   int = 256,
    lr:           float = 1e-3,
    channels:     int = 128,
    n_blocks:     int = 6,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pretrain] Using device: {device}")

    # Build dataset from n-tuple games
    boards, policies, values = generate_pretrain_dataset(
        ntuple_path, n_games=n_games, device=device
    )

    dataset    = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = PolicyValueNet(channels=channels, n_blocks=n_blocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    value_loss_fn  = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    print(f"[Pretrain] Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss   = 0.0
        total_vloss  = 0.0
        total_ploss  = 0.0
        n_batches    = 0

        for board_batch, policy_batch, value_batch in dataloader:
            board_batch  = board_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch  = value_batch.to(device)

            policy_logits, value_pred = model(board_batch)

            v_loss = value_loss_fn(value_pred, value_batch)
            p_loss = policy_loss_fn(policy_logits, policy_batch)
            loss   = v_loss + p_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_vloss += v_loss.item()
            total_ploss += p_loss.item()
            n_batches   += 1

        avg_loss  = total_loss  / n_batches
        avg_vloss = total_vloss / n_batches
        avg_ploss = total_ploss / n_batches
        print(
            f"  Epoch {epoch+1:>3}/{epochs} | "
            f"loss={avg_loss:.4f} | "
            f"value={avg_vloss:.4f} | "
            f"policy={avg_ploss:.4f}"
        )

    # Save model weights and config so Layer 4 can load it
    torch.save({
        'model_state_dict': model.state_dict(),
        'channels': channels,
        'n_blocks':  n_blocks,
    }, save_path)
    print(f"[Pretrain] Model saved to {save_path}")
    return model


if __name__ == "__main__":
    run_pretrain()