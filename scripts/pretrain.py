import os
import glob
from training.pretrain import run_pretrain

def find_emergency_checkpoint(save_path: str):
    pattern = save_path.replace('.pt', '_epoch*_emergency.pt')
    matches = sorted(glob.glob(pattern))
    if matches:
        print(f"[Resume] Found emergency checkpoint: {matches[-1]}")
        return matches[-1]
    return None

SAVE_PATH = "saved/models/pretrained_q.pt"
resume    = find_emergency_checkpoint(SAVE_PATH)

run_pretrain(
    demo_path   = "saved/demos/expert_demos_better.npz",
    save_path   = SAVE_PATH,
    log_path    = "saved/pretrain_log.csv",
    resume_path = resume,
    epochs      = 20,
    batch_size  = 512,
    lr          = 1e-3,
    channels    = 128,
    n_blocks    = 6,
    log_every   = 50,
)