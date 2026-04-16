from data.generator import generate_expert_demos

generate_expert_demos(
    save_path   = "saved/demos/expert_demos.npz",
    n_games     = 2000,
    white_depth = 3,
    black_depth = 3,
    time_limit  = 1.0,
    log_every   = 50,
    resume      = True,
)