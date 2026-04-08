import torch
import numpy as np
from reversi_ai.reversi import reversi as reversi_game
from model.network import PolicyValueNet
from search.mcts import MonteCarloTreeSearch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MCTS Test] Using device: {device}")

    # Load pretrained network
    checkpoint = torch.load(
        "data/checkpoints/pretrained.pt",
        map_location=device
    )
    model = PolicyValueNet(
        channels = checkpoint['channels'],
        n_blocks = checkpoint['n_blocks'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("[MCTS Test] Model loaded successfully")

    # Run MCTS from the starting position
    game = reversi_game()
    mcts = MonteCarloTreeSearch(model, device, c_puct=1.5)

    print("[MCTS Test] Running 400 simulations from opening position...")
    policy, value = mcts.search(
        board         = game.board,
        player        = game.turn,
        n_simulations = 400,
        add_noise     = False,
    )

    print(f"[MCTS Test] Root value estimate: {value:.4f}")
    print(f"[MCTS Test] Policy sum: {policy.sum():.4f} (should be 1.0)")

    # Show top 5 moves by visit count
    top_indices = np.argsort(policy)[::-1][:5]
    print("[MCTS Test] Top 5 moves by visit count:")
    for idx in top_indices:
        if policy[idx] > 0:
            x, y = idx // 8, idx % 8
            print(f"  ({x}, {y}) → probability {policy[idx]:.4f}")

    # Select an action
    action = mcts.select_action(policy, temperature=1.0)
    print(f"[MCTS Test] Selected action: {action}")
    print("[MCTS Test] Layer 4 working correctly.")

if __name__ == "__main__":
    main()

# helps generate moves during training 
# Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for making optimal
# decisions, particularly in complex artificial intelligence, games (like Go, chess), 
# and decision-making problems

#The tree is often kept in memory between moves. By pruning the tree to the new root node, the AI exploits 
# the statistical evidence gathered in previous turns, allowing it to refine its understanding of the game 
# as it progresses. 

#"Go-Exploit" is a reinforcement learning (RL) algorithm and search control strategy designed to improve the sample
#  efficiency and training speed of AlphaZero-based agents. It works by creating an archive of previously visited
#  "states of interest" and starting self-play training games from these varied positions, rather than always
#  starting from the beginning of a game (initial state)