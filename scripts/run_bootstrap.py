from bootstrap.td_trainer import run_bootstrap

run_bootstrap(
    n_games=50_000,
    lr=0.001,
    epsilon_start=0.3,
    epsilon_end=0.05,
    save_path="data/checkpoints/ntuple_weights.pk1",
    log_every=500,
)

## run with - python scripts/run_bootstrap.py

# Layer 2 is the bootstrapping phase that trains a lightweight n-tuple network using 
# Temporal Difference learning, entirely on CPU, before the expensive ResNet training 
# begins. It plays tens of thousands of self-play games using the existing `reversi.py` 
# engine, and after each game it works backwards through the move history nudging the 
# network's weight estimates toward the actual game outcome. The purpose is to produce a 
# value function that already has a reasonable understanding of which board positions are 
# winning and losing, so that when the ResNet in Layer 3 is initialized it has meaningful 
# targets to learn from rather than starting from random noise. The output of Layer 2 is a
#  single `ntuple_weights.pkl` file that Layer 3 reads once during supervised pretraining 
# and then never needs again.