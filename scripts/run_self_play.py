from training.self_play import run_self_play

run_self_play(
    checkpoint_path = "data/checkpoints/pretrained.pt",
    buffer_path     = "data/replay_buffer/buffer.npz",
    n_games         = 200,
    n_simulations   = 400,
    c_puct          = 1.5,
    temp_threshold  = 12,
    log_every       = 10,
)

# run n_games of self and save all positions to the replay buffer,