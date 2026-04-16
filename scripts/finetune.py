import os
import glob
from training.dqfd_trainer import run_finetuning

def find_latest(save_path: str):
    pattern = save_path.replace('.pt', '_game*_emergency.pt')
    matches = sorted(glob.glob(pattern))
    if matches:
        print(f"[Resume] Found emergency checkpoint: {matches[-1]}")
        return matches[-1]
    return None

SAVE_PATH = "saved/models/final_q.pt"
resume    = find_latest(SAVE_PATH)

run_finetuning(
    pretrained_path = "saved/models/pretrained_q.pt",
    demo_path       = "saved/demos/expert_demos_better.npz",  # ← correct
    save_path       = "saved/models/final_q.pt",
    log_path        = "saved/finetune_log.csv",
    resume_path     = None,
    n_games         = 500,
    batch_size      = 512,
    lr              = 5e-4,
    update_every    = 4,
    target_update   = 50,
    epsilon_start   = 0.3,
    epsilon_end     = 0.05,
    expert_fraction = 0.3,
    log_every       = 50,
    opponent_depth  = 1,
)