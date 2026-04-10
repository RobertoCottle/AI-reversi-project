import os
import csv
import torch
import numpy as np
import time
import shutil
import glob

from model.network import PolicyValueNet
from training.self_play import SelfPlayWorker
from training.replay_store import ReplayStore
from training.trainer import Trainer

DRIVE_BASE   = '/content/drive/MyDrive/AI-reversi-project'
PROJECT_BASE = '/content/AI-reversi-project'

def sync_to_drive():
    """Copy all checkpoints and buffer to Google Drive."""
    os.makedirs(f'{DRIVE_BASE}/data/checkpoints',   exist_ok=True)
    os.makedirs(f'{DRIVE_BASE}/data/replay_buffer', exist_ok=True)
    os.makedirs(f'{DRIVE_BASE}/data/training_logs', exist_ok=True)

    # Sync all checkpoint files
    for f in glob.glob(f'{PROJECT_BASE}/data/checkpoints/*.pt'):
        dst = f'{DRIVE_BASE}/data/checkpoints/{os.path.basename(f)}'
        shutil.copy(f, dst)

    # Sync replay buffer
    buf = f'{PROJECT_BASE}/data/replay_buffer/buffer.npz'
    if os.path.exists(buf):
        shutil.copy(buf, f'{DRIVE_BASE}/data/replay_buffer/buffer.npz')

    # Sync training log
    log = f'{PROJECT_BASE}/data/training_logs/training.csv'
    if os.path.exists(log):
        shutil.copy(log, f'{DRIVE_BASE}/data/training_logs/training.csv')

    print(f"  [Drive] Synced to Google Drive")

print("sync_to_drive() ready")



def run_training_loop(
    checkpoint_path:  str   = "data/checkpoints/pretrained.pt",
    buffer_path:      str   = "data/replay_buffer/buffer.npz",
    log_path:         str   = "data/training_logs/training.csv",
    best_path:        str   = "data/checkpoints/best.pt",
    n_iterations:     int   = 200,
    games_per_iter:   int   = 20,
    n_simulations:    int   = 400,
    update_steps:     int   = 100,
    batch_size:       int   = 256,
    min_buffer_size:  int   = 512,
    c_puct:           float = 1.5,
    temp_threshold:   int   = 12,
    lr:               float = 2e-3,
    l2_reg:           float = 1e-4,
    channels:         int   = 128,
    n_blocks:         int   = 6,
    save_every:       int   = 10,
):
    #Main alternating training loop.

    #Each iteration:
    #    1. Play games_per_iter self-play games → push to replay buffer
    #    2. Run update_steps gradient updates   → improve the network
    #    3. Save checkpoint every save_every iterations

    #The network used for self-play is updated in-place after each
    #gradient phase, so each iteration's games come from a slightly
    #stronger network than the last.
    os.makedirs("data/checkpoints",   exist_ok=True)
    os.makedirs("data/training_logs", exist_ok=True)
    os.makedirs("data/replay_buffer", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using device: {device}")

    # Load starting network
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PolicyValueNet(
        channels = channels,
        n_blocks = n_blocks,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_iteration = checkpoint.get('iteration', 0)
    print(f"[Training] Resuming from iteration {start_iteration}")

    # Load replay buffer
    store = ReplayStore(max_size=500_000, save_path=buffer_path)
    store.load()
    print(f"[Training] Replay buffer: {len(store):,} existing positions")

    # Initialize self-play worker and trainer
    # Both share the same model instance — when trainer updates weights,
    # the worker immediately uses the new weights for its next game
    worker = SelfPlayWorker(
        model          = model,
        device         = device,
        n_simulations  = n_simulations,
        c_puct         = c_puct,
        temp_threshold = temp_threshold,
        add_noise      = True,
    )

    trainer = Trainer(model=model, device=device, lr=lr, l2_reg=l2_reg)

    # Set up CSV log
    log_exists = os.path.exists(log_path)
    log_file   = open(log_path, 'a', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=[
        'iteration', 'buffer_size', 'white_wins', 'black_wins', 'draws',
        'loss', 'policy_loss', 'value_loss',
    ])
    if not log_exists:
        log_writer.writeheader()

    print(f"[Training] Starting loop: {n_iterations} iterations")
    print(f"           {games_per_iter} games/iter, {update_steps} update steps/iter")
    print(f"           batch={batch_size}, lr={lr}, sims={n_simulations}")
    print("-" * 65)

    best_loss = np.inf
    try:
        for iteration in range(start_iteration + 1, start_iteration + n_iterations + 1):
            iter_start = time.time()

            # ── Phase 1: Self-play ─────────────────────────────────────────
            white_wins = 0
            black_wins = 0
            draws      = 0

            model.eval()
            for _ in range(games_per_iter):
                trajectory, outcome = worker.play_one_game()
                store.push_game(trajectory)
                if outcome > 0:
                    white_wins += 1
                elif outcome < 0:
                    black_wins += 1
                else:
                    draws += 1

            # ── Phase 2: Gradient updates ──────────────────────────────────
            if len(store) < min_buffer_size:
                print(f"  Iter {iteration:>4} | Buffer too small ({len(store)}), skipping update")
                continue

            metrics = trainer.run_update_steps(store, update_steps, batch_size)

            # ── Logging ────────────────────────────────────────────────────
            total = games_per_iter
            iter_time = time.time() - iter_start
            print(
                f"  Iter {iteration:>4}/{start_iteration + n_iterations} | "
                f"Buf={len(store):>7,} | "
                f"W={white_wins/total:.2%} B={black_wins/total:.2%} D={draws/total:.2%} | "
                f"loss={metrics['loss']:.4f} "
                f"(p={metrics['policy_loss']:.4f} v={metrics['value_loss']:.4f}) | "
                f"time={iter_time:.1f}s"
            )

            log_writer.writerow({
                'iteration':   iteration,
                'buffer_size': len(store),
                'white_wins':  white_wins,
                'black_wins':  black_wins,
                'draws':       draws,
                'loss':        metrics['loss'],
                'policy_loss': metrics['policy_loss'],
                'value_loss':  metrics['value_loss'],
            })
            log_file.flush()

            # ── Checkpointing ──────────────────────────────────────────────
            if iteration % save_every == 0:
                iter_path = f"data/checkpoints/iter_{iteration:04d}.pt"
                trainer.save_checkpoint(iter_path, iteration, channels, n_blocks)
                print(f"           Checkpoint saved → {iter_path}")
                # Sync to Drive if running in Colab
                try:
                    sync_to_drive()
                except NameError:
                    pass   # sync_to_drive not defined outside Colab, skip silently

            # Always keep a copy of the best (lowest loss) model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                trainer.save_checkpoint(best_path, iteration, channels, n_blocks)
        store.save()
        trainer.save_checkpoint(best_path, iteration, channels, n_blocks)
        print("-" * 65)
        print(f"[Training] Complete. Best model saved to {best_path}")
        print(f"[Training] Final buffer size: {len(store):,} positions")
    #except 
    except KeyboardInterrupt:
        print("\n[Training] Interrupted. Saving emergency checkpoint...")
        emergency_path = f"data/checkpoints/iter_{iteration:04d}_emergency.pt"
        trainer.save_checkpoint(emergency_path, iteration, channels, n_blocks)
        store.save()
        try:
            sync_to_drive()
        except NameError:
            pass
        print(f"[Training] Saved to {emergency_path}")
        print(f"[Training] Buffer saved. Safe to exit.")
    #finally
    finally:
        log_file.close()

    # Final save
    #store.save()
    #trainer.save_checkpoint(best_path, iteration, channels, n_blocks)
    #log_file.close()

    # print("-" * 65)
    # print(f"[Training] Complete. Best model saved to {best_path}")
    # print(f"[Training] Final buffer size: {len(store):,} positions")