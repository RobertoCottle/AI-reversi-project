import numpy as np
import torch
from reversi_ai.reversi import reversi as reversi_game
from bootstrap.td_selfplay import get_legal_moves
from model.encoder import encode_board
from model.network import PolicyValueNet
from search.mcts import MonteCarloTreeSearch


class NeuralAgent:
    
    #Wraps PolicyValueNet + MCTS for use in evaluation matches.
    #Uses a lower simulation count than training for faster evaluation.
    

    def __init__(
        self,
        model:         PolicyValueNet,
        device:        torch.device,
        n_simulations: int   = 200,
        c_puct:        float = 1.5,
    ):
        self.mcts = MonteCarloTreeSearch(model, device, c_puct)
        self.n_simulations = n_simulations

    def select_action(self, board: np.ndarray, player: int):
        policy, _ = self.mcts.search(
            board         = board,
            player        = player,
            n_simulations = self.n_simulations,
            add_noise     = False,   # no noise during evaluation
        )
        return self.mcts.select_action(policy, temperature=0.0)

def play_match(agent_white, agent_black, debug=False) -> int:
    game = reversi_game()
    consecutive_passes = 0
    move_count = 0

    while True:
        player = game.turn
        agent  = agent_white if player == 1 else agent_black
        legal  = get_legal_moves(game, player)

        if not legal:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.turn = -game.turn
            continue
        else:
            consecutive_passes = 0

        action = agent.select_action(game.board, player)
        x, y   = action
        move_count += 1

        if x == -1 and y == -1:
            game.turn = -game.turn
        else:
            game.step(x, y, player, commit=True)
            game.turn = -game.turn

    white = int(np.sum(game.board == 1))
    black = int(np.sum(game.board == -1))

    if debug:
        print(f"  Game over after {move_count} moves | White={white} Black={black}")

    if white > black:
        return 1
    elif black > white:
        return -1
    return 0

# def play_match(
#     agent_white,
#     agent_black,
# ) -> int:
    
#     #Play one complete game between two agents.

#     #agent_white plays as white (player=1, moves first).
#     #agent_black plays as black (player=-1).

#     #Returns:
#     #     1 if white wins
#     #    -1 if black wins
#     #     0 if draw
    
#     game = reversi_game()
#     consecutive_passes = 0
#     #move_count = 0 #for debug

#     while True:
#         player = game.turn
#         agent  = agent_white if player == 1 else agent_black #this might be the problem
#         legal = get_legal_moves(game, player)

#         if not legal:
#             consecutive_passes += 1
#             if consecutive_passes >= 2:
#                 break
#             game.turn = -game.turn
#             continue
#         else:
#             consecutive_passes = 0

#         action = agent.select_action(game.board, player)
#         x, y   = action
#         #move_count += 1

#          # Temporary debug — remove after diagnosis
#         #if move_count <= 3:
#         #    print(f"  move {move_count}: player={player} action={action} legal={legal[:3]}")


#         if x == -1 and y == -1:
#             game.turn = -game.turn
#         else:
#             #result = game.step(x, y, player, commit=True)
#             # Temporary debug
#             #if move_count <= 3:
#             #    print(f"  step result: {result}")
#             #game.turn = -game.turn
#             game.step(x, y, player, commit=True)
#             game.turn = -game.turn

#     white = int(np.sum(game.board == 1))
#     black = int(np.sum(game.board == -1))

#     if white > black:
#         return 1
#     elif black > white:
#         return -1
#     return 0


def run_evaluation(
    checkpoint_path: str   = "data/checkpoints/best.pt",
    n_games:         int   = 40,
    n_simulations:   int   = 200,
    c_puct:          float = 1.5,
):

    #Evaluate the trained agent against the full opponent ladder.

    #Plays n_games against each opponent (half as white, half as black)
    #and reports win rates.

    #Args:
    #    checkpoint_path: path to the model checkpoint to evaluate
    #    n_games:         total games per opponent (must be even)
    #    n_simulations:   MCTS simulations per move during evaluation
    #    c_puct:          exploration constant
    
    from evaluation.opponents import RandomAgent, GreedyAgent, AlphaBetaAgent

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluation] Using device: {device}")
    print(f"[Evaluation] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PolicyValueNet(
        channels = checkpoint['channels'],
        n_blocks = checkpoint['n_blocks'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    neural_agent = NeuralAgent(model, device, n_simulations, c_puct)

    opponents = [
        ("Random",        RandomAgent()),
        ("Greedy",        GreedyAgent()),
        ("AlphaBeta-d1",  AlphaBetaAgent(depth=1)),
        ("AlphaBeta-d3",  AlphaBetaAgent(depth=3)),
        ("AlphaBeta-d5",  AlphaBetaAgent(depth=5)),
    ]

    half = n_games // 2
    print(f"\n[Evaluation] {n_games} games per opponent ({half} as white, {half} as black)")
    print("-" * 60)
    print(f"  {'Opponent':<16} {'W':>5} {'L':>5} {'D':>5} {'Win%':>7}")
    print("-" * 60)

    results = {}

    for name, opponent in opponents:
        wins = losses = draws = 0

        # Play half the games as white (our agent moves first)
        for _ in range(half):
            #outcome = play_match(neural_agent, opponent)
            outcome = play_match(neural_agent, opponent, debug=True)
            if outcome == 1:
                wins += 1
            elif outcome == -1:
                losses += 1
            else:
                draws += 1

        # Play half the games as black (opponent moves first)
        for _ in range(half):
            #outcome = play_match(opponent, neural_agent)
            outcome = play_match(neural_agent, opponent, debug=True)
            if outcome == -1:    # we are black, -1 means black wins
                wins += 1
            elif outcome == 1:
                losses += 1
            else:
                draws += 1

        win_rate = wins / n_games
        results[name] = {'wins': wins, 'losses': losses,
                         'draws': draws, 'win_rate': win_rate}

        print(f"  {name:<16} {wins:>5} {losses:>5} {draws:>5} {win_rate:>7.1%}")

    print("-" * 60)
    print(f"\n[Evaluation] Done. Checkpoint: {checkpoint_path}")
    return results