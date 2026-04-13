#single game trace
# import torch
# import numpy as np
# from reversi_ai.reversi import reversi as reversi_game
# from bootstrap.td_selfplay import get_legal_moves
# from model.network import PolicyValueNet
# from model.encoder import encode_board
# from search.mcts import MonteCarloTreeSearch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint = torch.load("data/checkpoints/best.pt", map_location=device)
# model = PolicyValueNet(
#     channels = checkpoint['channels'],
#     n_blocks = checkpoint['n_blocks'],
# ).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# mcts = MonteCarloTreeSearch(model, device, c_puct=1.5)
# game = reversi_game()
# consecutive_passes = 0
# move_count = 0

# print("Move by move trace (neural=white, random=black):")
# print(f"Start: White={int(np.sum(game.board==1))} Black={int(np.sum(game.board==-1))}")

# while move_count < 10:
#     player = game.turn
#     legal  = get_legal_moves(game, player)

#     if not legal:
#         consecutive_passes += 1
#         if consecutive_passes >= 2:
#             break
#         game.turn = -game.turn
#         continue
#     else:
#         consecutive_passes = 0

#     if player == 1:
#         # Neural agent plays as white
#         policy, value = mcts.search(
#             board=game.board,
#             player=player,
#             n_simulations=50,
#             add_noise=False,
#         )
#         action = mcts.select_action(policy, temperature=0.0)
#         agent_name = "Neural"
#     else:
#         # Random agent plays as black
#         action = legal[np.random.randint(len(legal))]
#         agent_name = "Random"

#     x, y = action
#     result = game.step(x, y, player, commit=True)
#     game.turn = -game.turn
#     move_count += 1

#     print(f"  Move {move_count}: {agent_name} ({'W' if player==1 else 'B'}) "
#           f"plays ({x},{y}) → flipped {result} | "
#           f"White={int(np.sum(game.board==1))} "
#           f"Black={int(np.sum(game.board==-1))}")
#----------------------------------

# import torch
# import numpy as np
# from reversi_ai.reversi import reversi as reversi_game
# from model.network import PolicyValueNet
# from model.encoder import encode_board
# from bootstrap.td_selfplay import get_legal_moves

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checkpoint = torch.load("data/checkpoints/best.pt", map_location=device)
# model = PolicyValueNet(
#     channels = checkpoint['channels'],
#     n_blocks = checkpoint['n_blocks'],
# ).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Test 1: what does the network output on the opening position?
# game = reversi_game()
# tensor = encode_board(game.board, game.turn)
# policy, value = model.predict(tensor, device)

# print(f"Opening position value estimate: {value:.4f}")
# print(f"Policy sum: {policy.sum():.4f}")
# print(f"Policy max: {policy.max():.4f}")
# print(f"Policy min: {policy.min():.4f}")
# print(f"Pass action probability: {policy[64]:.4f}")
# print(f"\nTop 5 moves:")
# top = np.argsort(policy)[::-1][:5]
# for idx in top:
#     if policy[idx] > 0.001:
#         x, y = idx // 8, idx % 8
#         print(f"  ({x},{y}) = {policy[idx]:.4f}")

# # Test 2: are the legal moves getting nonzero probability?
# legal = get_legal_moves(game, game.turn)
# print(f"\nLegal moves: {legal}")
# legal_probs = [policy[x*8+y] for x,y in legal]
# print(f"Legal move probabilities: {legal_probs}")
# print(f"Illegal move total probability: {sum(policy[i] for i in range(64) if (i//8, i%8) not in legal and i != 64):.4f}")

# ckpt = torch.load("data/checkpoints/best.pt", map_location='cpu')
# print(f"best.pt iteration: {ckpt.get('iteration', 'unknown')}")

# 3 game trace
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
for i in range(5):
    outcome = play_match(neural, random, debug=False)
    print(f"  Game {i+1}: {'+1 white wins' if outcome==1 else '-1 black wins' if outcome==-1 else 'draw'}")

print("\nTesting random (white) vs neural (black):")
for i in range(5):
    outcome = play_match(random, neural, debug=False)
    print(f"  Game {i+1}: {'+1 white wins' if outcome==1 else '-1 black wins' if outcome==-1 else 'draw'}")

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