from data.generate_from_better_player import generate_expert_demos_better

generate_expert_demos_better(
    save_path = "saved/demos/expert_demos_better.npz",
    n_games   = 2000,
    log_every = 50,
    resume    = True,
    configs   = [
        ('better',    'alphabeta', 200),
        ('alphabeta', 'better',    200),
        ('better',    'greedy',    200),
        ('greedy',    'better',    200),
        ('greedy',    'random',    200),
        ('random',    'greedy',    200),
        ('greedy',    'alphabeta', 200),
        ('alphabeta', 'greedy',    200),
        ('random',    'alphabeta', 200),
        ('alphabeta', 'random',    200),
    ],
)

# from data.generate_from_better_player import generate_expert_demos_better

# generate_expert_demos_better(
#     save_path = "saved/demos/expert_demos_better.npz",
#     n_games   = 2000,
#     log_every = 50,
#     resume    = True,
#     configs   = [
#         ('better', 'greedy', 1000),  # better as white vs greedy black
#         ('greedy', 'better', 1000),  # greedy white vs better as black
#         #('better', 'better', 1000),  # best vs best, color-swapped
#         #('better', 'random',  500),  # better as white
#         #('random', 'better',  500),  # better as black
#     ],
# )

# #  