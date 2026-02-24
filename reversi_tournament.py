import sys 
import importlib
import numpy as np
from reversi import reversi

NUM_GAMES = 100

# --------------------
# Utility Functions
# --------------------

#gets function from module that returns var: tuple[Literal[-1], Literal[-1]] | tuple[int, int]
# og function must specify game = reversi() game.board = board.copy() within scope
def get_move_func(module):
    if hasattr(module, "get_tourn_move"):
        return getattr(module, "get_tourn_move")
    elif hasattr(module, "choose_move"):
        return getattr(module, "choose_move")
    else:
        raise AttributeError(f"Module '{module.__name__}' has neither 'get_tourn_move' nor 'choose_move'.")

def get_phase(move_number):
    if move_number < 20:
        return "early"
    elif move_number < 45:
        return "mid"
    else:
        return "late"
    
def count_corners(board):
    corners = [(0,0), (0,7), (7,0), (7,7)]
    corner_count = { -1: 0, 1: 0}

    for x, y in corners:
        if board[x][y] != 0:
            corner_count[board[x][y]] += 1

    return corner_count

def play_game(player1, player2):
    game = reversi()
    turn = 1 #white player starts
    passes = 0
    move_number = 0

    phase_differentials = {
        "early" : [],
        "mid": [],
        "late": []
    }

    player1_move = get_move_func(player1)
    player2_move = get_move_func(player2)

    while True:
        # if turn == 1:
        #     move = player1.get_tourn_move(game.board.copy(), turn)
        # else:
        #     move = player2.get_tourn_move(game.board.copy(), turn)

        if turn == 1:
            move = player1_move(game.board.copy(), turn)
        else:
            move = player2_move(game.board.copy(), turn)

        if move == (-1,-1): 
            passes += 1 #passes increase if no legal moves available 
        else:
            game.step(move[0], move[1], turn, True)
            passes = 0

            # Track phase performance
            phase = get_phase(move_number)
            white_count = np.sum(game.board == 1)
            black_count = np.sum(game.board == -1)
            diff = white_count - black_count
            phase_differentials[phase].append(diff)
        
        if passes == 2:
            break # both players passed -> game over

        turn *= -1
        move_number += 1

    return game.board, phase_differentials
    
    # ------------
    # Tournament
    # ------------

def run_tournament(player1, player2):
    results = {
    "black_wins": 0,
    "white_wins": 0,
    "draws": 0,
    "black_discs": [],
    "white_discs": [],
    "corner_stats": [],
    "phase_stats": {
        "early": [],
        "mid": [],
        "late": []
    }
    }
        
    for i in range(NUM_GAMES):

        #Alternate starting player
        if i % 2 == 0:
            board, phase_data = play_game(player1, player2)
        else:
            board, phase_data = play_game(player2, player1)

        black_count = np.sum(board == -1)
        white_count = np.sum(board == 1)

        results["black_discs"].append(black_count)
        results["white_discs"].append(white_count)

        # win tracking
        if black_count > white_count:
            results["black_wins"] += 1
        elif white_count > black_count:
            results["white_wins"] += 1
        else:
            results["draws"] += 1

        # Corner tracking
        results["corner_stats"].append(count_corners(board))

        # phase tracking
        for phase in ["early", "mid", "late"]:
            if phase_data[phase]:
                avg_phase_diff = np.mean(phase_data[phase])
                results['phase_stats'][phase].append(avg_phase_diff)
    
    return results

def print_results(results):

    print("\n===== TOURNAMENT RESULTS =====")
    print(f"Black Wins: {results['black_wins']}")
    print(f"White Wins: {results['white_wins']}")
    print(f"Draws: {results['draws']}")

    print("\nAverage Discs:")
    print(f"Black Avg: {np.mean(results['black_discs']):.2f}")
    print(f"White Avg: {np.mean(results['white_discs']):.2f}")

    black_corner_total = sum(game[-1] for game in results["corner_stats"])
    white_corner_total = sum(game[1] for game in results["corner_stats"])

    print("\nTotal Corners Controlled:")
    print(f"Black Corners: {black_corner_total}")
    print(f"White Corners: {white_corner_total}")

    print("\nPhase Performance (White Disc Advantage Avg):")
    for phase in ["early", "mid", "late"]:
        if results["phase_stats"][phase]:
            print(f"{phase.capitalize()}: {np.mean(results['phase_stats'][phase]):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reversi_tournament.py player1 player2")
        sys.exit(1)

    # Import players dynamically
    player1 = importlib.import_module(sys.argv[1])
    player2 = importlib.import_module(sys.argv[2])

    results = run_tournament(player1, player2)
    print_results(results)

## this is how it accepts the command line argument
# python reversi_tournament.py greedy_player greedy_player