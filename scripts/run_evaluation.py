import torch
import numpy as np
from reversi_ai.reversi import reversi as reversi_game
from model.network import PolicyValueNet
from evaluation.evaluator import NeuralAgent, play_match
from evaluation.opponents import RandomAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("data/checkpoints/iter_0200.pt", map_location=device)
model = PolicyValueNet(
    channels = checkpoint['channels'],
    n_blocks = checkpoint['n_blocks'],
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

neural = NeuralAgent(model, device, n_simulations=50)
random = RandomAgent()

print("Testing neural (white) vs random (black):")
for i in range(3):
    outcome = play_match(neural, random, debug=True)
    print(f"  Game {i+1} outcome: {outcome} (1=white wins, -1=black wins)")

print("\nTesting random (white) vs neural (black):")
for i in range(3):
    outcome = play_match(random, neural, debug=True)
    print(f"  Game {i+1} outcome: {outcome} (1=white wins, -1=black wins)")

# import torch
# from evaluation.evaluator import run_evaluation
# from evaluation.metrics import save_evaluation_results, print_summary

# ckpt = torch.load("data/checkpoints/best.pt", map_location='cpu')
# print(f"best.pt was saved at iteration: {ckpt.get('iteration', 'unknown')}")

# results = run_evaluation(
#     checkpoint_path = "data/checkpoints/iter_0200.pt",
#     n_games         = 10,
#     n_simulations   = 200,
#     c_puct          = 1.5,
#)

# results = run_evaluation(
#     checkpoint_path = "data/checkpoints/best.pt",
#     n_games         = 40,
#     n_simulations   = 200,
#     c_puct          = 1.5,
# )

#save_evaluation_results(results, "data/checkpoints/best.pt")
#print_summary(results)

#run script

# python scripts/run_evaluation.py