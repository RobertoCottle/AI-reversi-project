import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves, encode_board
from network.q_network import QNetwork
from data.demo_buffer import DemoBuffer
from agents.alphabeta_agent import select_move_timed
from training.pretrain import supervised_loss


class ReplayBuffer:
    """Combined buffer holding both expert demos and self-play transitions."""

    def __init__(self, max_size: int = 200_000):
        self.buffer   = deque(maxlen=max_size)
        self.max_size = max_size

    def push(self, board, action, reward, next_board, done, is_expert=False):
        self.buffer.append((board, action, reward, next_board, done, is_expert))

    def sample(self, batch_size: int, expert_fraction: float = 0.3) -> dict:
        """
        Sample a batch with a guaranteed fraction of expert transitions.
        This prevents the model from forgetting expert knowledge.
        """
        expert  = [t for t in self.buffer if t[5]]
        self_play = [t for t in self.buffer if not t[5]]

        n_expert   = min(int(batch_size * expert_fraction), len(expert))
        n_self     = min(batch_size - n_expert, len(self_play))
        n_expert   = min(batch_size - n_self, len(expert))

        batch = []
        if n_expert > 0:
            idx   = np.random.choice(len(expert), n_expert, replace=False)
            batch += [expert[i] for i in idx]
        if n_self > 0:
            idx   = np.random.choice(len(self_play), n_self, replace=False)
            batch += [self_play[i] for i in idx]

        boards      = np.stack([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch])
        rewards     = np.array([t[2] for t in batch], dtype=np.float32)
        next_boards = np.stack([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch], dtype=np.float32)
        is_expert   = np.array([t[5] for t in batch])

        return {
            'boards':      boards,
            'actions':     actions,
            'rewards':     rewards,
            'next_boards': next_boards,
            'dones':       dones,
            'is_expert':   is_expert,
        }

    def __len__(self):
        return len(self.buffer)


def run_finetuning(
    pretrained_path: str   = "saved/models/pretrained_q.pt",
    demo_path:       str   = "saved/demos/expert_demos.npz",
    save_path:       str   = "saved/models/final_q.pt",
    log_path:        str   = "saved/finetune_log.csv",
    resume_path:     str   = None,
    n_games:         int   = 500,
    batch_size:      int   = 512,
    lr:              float = 5e-4,
    update_every:    int   = 4,
    target_update:   int   = 50,
    epsilon_start:   float = 0.3,
    epsilon_end:     float = 0.05,
    expert_fraction: float = 0.3,
    log_every:       int   = 50,
    opponent_depth:  int   = 1,
):
    """
    Phase 3: Fine-tune via self-play against AlphaBeta.

    Uses epsilon-greedy exploration against AlphaBeta-d1
    while keeping expert demos in the replay buffer.

    Supports Ctrl+C and resume.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path),  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Finetune] Using device: {device}")

    # Load model
    load_from = resume_path if (resume_path and os.path.exists(resume_path)) \
                else pretrained_path
    model = QNetwork.load(load_from, device)
    print(f"[Finetune] Loaded from {load_from}")

    target_net = QNetwork(
        channels = model.channels,
        n_blocks = model.n_blocks,
    ).to(device)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    td_loss_fn = nn.MSELoss()

    # Build replay buffer — seed with expert demos
    replay = ReplayBuffer(max_size=200_000)
    demo_buf = DemoBuffer(demo_path)
    if demo_buf.load():
        for i in range(len(demo_buf)):
            replay.push(
                demo_buf.boards[i], demo_buf.actions[i],
                demo_buf.rewards[i], demo_buf.next_boards[i],
                demo_buf.dones[i], is_expert=True
            )
        print(f"[Finetune] Seeded replay with "
              f"{len(demo_buf):,} expert transitions")

    # Load progress state if resuming
    progress_path = save_path.replace('.pt', '_progress.npz')
    start_game    = 0
    if resume_path and os.path.exists(progress_path):
        prog       = np.load(progress_path)
        start_game = int(prog['games_done'])
        print(f"[Finetune] Resuming from game {start_game}")

    # CSV log
    log_exists = os.path.exists(log_path)
    log_file   = open(log_path, 'a', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=[
        'game', 'outcome', 'epsilon', 'replay_size',
        'td_loss', 'sup_loss', 'elapsed_min'
    ])
    if not log_exists:
        log_writer.writeheader()

    wins        = {'agent': 0, 'opponent': 0, 'draw': 0}
    train_start = time.time()
    step        = 0
    best_win_rate = 0.0

    print(f"[Finetune] Fine-tuning {n_games - start_game} games "
          f"vs AlphaBeta-d{opponent_depth}")
    print(f"[Finetune] Epsilon: {epsilon_start} → {epsilon_end}")
    print(f"[Finetune] Expert fraction in batches: {expert_fraction:.0%}")
    print("-" * 60)

    try:
        for game_idx in range(start_game, n_games):
            epsilon = epsilon_start + \
                      (epsilon_end - epsilon_start) * (game_idx / n_games)

            outcome, transitions = _play_self_play_game(
                model, device, epsilon, opponent_depth
            )

            # Push transitions to replay buffer
            for (b, a, r, nb, done) in transitions:
                replay.push(b, a, r, nb, done, is_expert=False)

            if outcome == 1:
                wins['agent'] += 1
            elif outcome == -1:
                wins['opponent'] += 1
            else:
                wins['draw'] += 1

            # Training updates
            avg_td  = 0.0
            avg_sup = 0.0
            n_updates = 0

            if len(replay) >= batch_size:
                for _ in range(update_every):
                    batch = replay.sample(batch_size, expert_fraction)
                    td, sup = _update_step(
                        model, target_net, optimizer,
                        td_loss_fn, batch, device
                    )
                    avg_td  += td
                    avg_sup += sup
                    n_updates += 1
                    step += 1

                if step % target_update == 0:
                    target_net.load_state_dict(model.state_dict())

            games_done = game_idx - start_game + 1

            if (game_idx + 1) % log_every == 0:
                elapsed   = (time.time() - train_start) / 60
                rate      = games_done / (time.time() - train_start)
                remaining = (n_games - game_idx - 1) / rate / 60 \
                            if rate > 0 else 0
                win_rate  = wins['agent'] / games_done

                avg_td_log  = avg_td  / n_updates if n_updates else 0
                avg_sup_log = avg_sup / n_updates if n_updates else 0

                print(
                    f"  Game {game_idx+1:>5}/{n_games} | "
                    f"ε={epsilon:.3f} | "
                    f"W={win_rate:.1%} "
                    f"L={wins['opponent']/games_done:.1%} "
                    f"D={wins['draw']/games_done:.1%} | "
                    f"replay={len(replay):,} | "
                    f"td={avg_td_log:.4f} sup={avg_sup_log:.4f} | "
                    f"elapsed={elapsed:.1f}min ETA={remaining:.1f}min"
                )

                log_writer.writerow({
                    'game':        game_idx + 1,
                    'outcome':     outcome,
                    'epsilon':     f"{epsilon:.4f}",
                    'replay_size': len(replay),
                    'td_loss':     f"{avg_td_log:.4f}",
                    'sup_loss':    f"{avg_sup_log:.4f}",
                    'elapsed_min': f"{elapsed:.2f}",
                })
                log_file.flush()

                # Save best model by win rate
                if win_rate >= best_win_rate:
                    best_win_rate = win_rate
                    model.save(save_path)

                # Save progress for resume
                np.savez(progress_path, games_done=game_idx + 1)

    except KeyboardInterrupt:
        print(f"\n[Finetune] Interrupted at game {game_idx + 1}")
        emergency = save_path.replace('.pt', f'_game{game_idx+1}_emergency.pt')
        model.save(emergency)
        np.savez(progress_path, games_done=game_idx + 1)
        print(f"[Finetune] Emergency save: {emergency}")
        print(f"[Finetune] Resume with resume_path='{emergency}'")

    finally:
        log_file.close()

    print(f"\n[Finetune] Complete. Best model: {save_path}")
    print(f"[Finetune] Final win rate vs AlphaBeta-d{opponent_depth}: "
          f"{best_win_rate:.1%}")
    return model


def _update_step(model, target_net, optimizer, td_loss_fn, batch, device):
    model.train()

    boards      = torch.tensor(batch['boards'],      dtype=torch.float32).to(device)
    actions     = torch.tensor(batch['actions'],     dtype=torch.long).to(device)
    rewards     = torch.tensor(batch['rewards'],     dtype=torch.float32).to(device)
    next_boards = torch.tensor(batch['next_boards'], dtype=torch.float32).to(device)
    dones       = torch.tensor(batch['dones'],       dtype=torch.float32).to(device)
    is_expert   = torch.tensor(batch['is_expert'],   dtype=torch.bool).to(device)

    q_current = model(boards)

    with torch.no_grad():
        q_next    = target_net(next_boards).max(dim=1).values
        td_target = rewards + 0.99 * q_next * (1 - dones)

    q_taken  = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss_td  = td_loss_fn(q_taken, td_target)

    # Supervised loss only on expert transitions
    loss_sup = torch.tensor(0.0, device=device)
    if is_expert.any():
        loss_sup = supervised_loss(
            q_current[is_expert],
            actions[is_expert],
        )

    loss = loss_td + loss_sup

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_td.item(), loss_sup.item()


def _play_self_play_game(
    model:          QNetwork,
    device:         torch.device,
    epsilon:        float,
    opponent_depth: int,
) -> tuple:
    """
    Play one game: Q-network (white) vs AlphaBeta (black).
    Uses epsilon-greedy exploration.
    Returns (outcome, transitions).
    """
    game    = reversi_game()
    history = []
    consecutive_passes = 0

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

        board_before = game.board.copy()

        if player == 1:
            # Q-network plays as white with epsilon-greedy
            if np.random.random() < epsilon:
                idx  = np.random.randint(len(legal))
                x, y = legal[idx]
            else:
                x, y = model.predict_move(
                    game.board, player, legal, device
                )
        else:
            # AlphaBeta plays as black
            x, y = select_move_timed(
                game.board, player,
                time_limit = 1.0,
                max_depth  = opponent_depth,
            )

        if x == -1 and y == -1:
            action = 64
            game.turn = -game.turn
        else:
            action = x * 8 + y
            game.step(x, y, player, commit=True)
            game.turn = -game.turn

        next_board = game.board.copy()
        board_tensor      = encode_board(board_before, player).numpy()
        next_board_tensor = encode_board(next_board, -player).numpy()

        history.append((board_tensor, action, 0.0, next_board_tensor, False))

    white   = int(np.sum(game.board == 1))
    black   = int(np.sum(game.board == -1))
    outcome = 1.0 if white > black else -1.0 if black > white else 0.0

    # Assign terminal reward to last transition
    if history:
        b, a, _, nb, _ = history[-1]
        history[-1]     = (b, a, outcome, nb, True)

    return outcome, history