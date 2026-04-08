import os
import glob
from training.train_loop import run_training_loop

TOTAL_ITERATIONS = 200  # this is the absolute target, not additional n

def find_latest_checkpoint(checkpoint_dir: str = "data/checkpoints") -> str:
    pattern = os.path.join(checkpoint_dir, "iter_*.pt") 
    checkpoints = sorted(glob.glob(pattern))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[Resume] Found checkpoint: {latest}")
        return latest 
    else:
        print("[Resume] No iteration checkpoints found, starting from pretrained.pt")
        return os.path.join(checkpoint_dir, "pretrained.pt")

def get_remaining_iterations(checkpoint_path: str) -> int:
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    start = checkpoint.get('iteration', 0)
    remaining = max(0, TOTAL_ITERATIONS - start)
    print(f"[Resume] Completed {start}/{TOTAL_ITERATIONS} iterations, {remaining} remaining")
    return remaining

checkpoint = find_latest_checkpoint()
remaining  = get_remaining_iterations(checkpoint)

if remaining == 0:
    print("[Resume] Training already complete.")
else:
    run_training_loop(
        checkpoint_path = checkpoint,
        buffer_path     = "data/replay_buffer/buffer.npz",
        log_path        = "data/training_logs/training.csv",
        best_path       = "data/checkpoints/best.pt",
        n_iterations    = remaining,
        games_per_iter  = 20,
        n_simulations   = 100,
        update_steps    = 100,
        batch_size      = 256,
        min_buffer_size = 512,
        c_puct          = 1.5,
        temp_threshold  = 12,
        lr              = 2e-3,
        l2_reg          = 1e-4,
        channels        = 128,
        n_blocks        = 6,
        save_every      = 10,
    )



# import os
# import glob
# from training.train_loop import run_training_loop

# TOTAL_ITERATIONS = 200  # this is the absolute target, not additional

# def find_latest_checkpoint(checkpoint_dir: str = "data/checkpoints") -> str:
#     """
#     Find the most recent numbered checkpoint file.
#     Falls back to pretrained.pt if no iteration checkpoints exist yet.
#     """
#     pattern = os.path.join(checkpoint_dir, "iter_*.pt")
#     checkpoints = sorted(glob.glob(pattern))

#     if checkpoints:
#         latest = checkpoints[-1]
#         print(f"[Resume] Found checkpoint: {latest}")
#         return latest
#     else:
#         print("[Resume] No iteration checkpoints found, starting from pretrained.pt")
#         return os.path.join(checkpoint_dir, "pretrained.pt")

# run_training_loop(
#     checkpoint_path = find_latest_checkpoint(),
#     buffer_path     = "data/replay_buffer/buffer.npz",
#     log_path        = "data/training_logs/training.csv",
#     best_path       = "data/checkpoints/best.pt",
#     n_iterations    = 200,
#     games_per_iter  = 20,
#     n_simulations   = 100,
#     update_steps    = 100,
#     batch_size      = 256,
#     min_buffer_size = 512,
#     c_puct          = 1.5,
#     temp_threshold  = 12,
#     lr              = 2e-3,
#     l2_reg          = 1e-4,
#     channels        = 128,
#     n_blocks        = 6,
#     save_every      = 10,
# )

# # layer 6

# # Start training
# #python scripts/run_training.py

# # Stop anytime with Ctrl+C
# # Later, resume from where you left off
# #python scripts/run_training.py