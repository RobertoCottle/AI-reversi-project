from model.pretrain import run_pretrain

run_pretrain(
    ntuple_path = "data/checkpoints/ntuple_weights.pkl",
    save_path   = "data/checkpoints/pretrained.pt",
    n_games     = 5_000,
    epochs      = 20,
    batch_size  = 256,
    lr          = 1e-3,
    channels    = 128,
    n_blocks    = 6,
)

# python scripts/run_pretrain.py