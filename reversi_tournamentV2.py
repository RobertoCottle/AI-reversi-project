import sys
import importlib
from reversi import reversi
import numpy as np


# -----------------------------
# Load player function
# -----------------------------
def load_player(module_name):
    module = importlib.import_module(module_name)

    if hasattr(module, "get_tourn_move"):
        return module.get_tourn_move
    elif hasattr(module, "choose_move"):
        return module.choose_move
    else:
        raise AttributeError(
            f"{module_name} must define get_tourn_move or choose_move"
        )


# -----------------------------
# Count discs
# -----------------------------
def count_discs(board):
    white = np.sum(board == 1)
    black = np.sum(board == -1)
    return white, black


# -----------------------------
# Count corners
# -----------------------------
def count_corners(board):
    corners = [(0,0), (0,7), (7,0), (7,7)]
    white = 0
    black = 0

    for r,c in corners:
        if board[r][c] == 1:
            white += 1
        elif board[r][c] == -1:
            black += 1

    return white, black


# -----------------------------
# Phase classification
# -----------------------------
def get_phase(move_count):
    if move_count < 20:
        return "early"
    elif move_count < 45:
        return "mid"
    else:
        return "late"


# -----------------------------
# Play one game
# -----------------------------
def play_game(player1_func, player2_func):

    game = reversi()
    turn = 1   # 1 = white, -1 = black
    move_count = 0
    consecutive_passes = 0

    phase_corner_control = {
        "early": {"white": 0, "black": 0},
        "mid": {"white": 0, "black": 0},
        "late": {"white": 0, "black": 0},
    }

    while True:

        if turn == 1:
            move = player1_func(game.board.copy(), turn)
        else:
            move = player2_func(game.board.copy(), turn)

        if move == (-1, -1):
            consecutive_passes += 1

            if consecutive_passes == 2:
                break   # Game over

            turn *= -1
            continue

        # Valid move played
        consecutive_passes = 0
        game.step(move[0], move[1], turn, True)
        move_count += 1

        # Track phase stats
        phase = get_phase(move_count)
        w_c, b_c = count_corners(game.board)

        phase_corner_control[phase]["white"] = w_c
        phase_corner_control[phase]["black"] = b_c

        turn *= -1

    white_discs, black_discs = count_discs(game.board)

    if white_discs > black_discs:
        winner = 1
    elif black_discs > white_discs:
        winner = -1
    else:
        winner = 0

    final_white_corners, final_black_corners = count_corners(game.board)

    return {
        "winner": winner,
        "white_discs": white_discs,
        "black_discs": black_discs,
        "white_corners": final_white_corners,
        "black_corners": final_black_corners,
        "phase_stats": phase_corner_control
    }


# -----------------------------
# Tournament runner
# -----------------------------
def run_tournament(player1_name, player2_name, num_games):

    p1_func = load_player(player1_name)
    p2_func = load_player(player2_name)

    stats = {
        player1_name: {"wins": 0, "losses": 0, "win_discs": [], "loss_discs": [], "corner_wins": []},
        player2_name: {"wins": 0, "losses": 0, "win_discs": [], "loss_discs": [], "corner_wins": []}
    }

    phase_totals = {
        "early": {"white": 0, "black": 0},
        "mid": {"white": 0, "black": 0},
        "late": {"white": 0, "black": 0},
    }

    for i in range(num_games):

        # Alternate colors
        if i % 2 == 0:
            result = play_game(p1_func, p2_func)
            p1_color = 1
            p2_color = -1
        else:
            result = play_game(p2_func, p1_func)
            p1_color = -1
            p2_color = 1

        winner = result["winner"]

        # Aggregate phase stats
        for phase in ["early", "mid", "late"]:
            phase_totals[phase]["white"] += result["phase_stats"][phase]["white"]
            phase_totals[phase]["black"] += result["phase_stats"][phase]["black"]

        if winner != 0:

            if winner == p1_color:
                winner_name = player1_name
                loser_name = player2_name
            else:
                winner == p2_color
                winner_name = player2_name
                loser_name = player1_name

            winner_discs = result["white_discs"] if winner == 1 else result["black_discs"]
            loser_discs = result["black_discs"] if winner == 1 else result["white_discs"]
            winner_corners = result["white_corners"] if winner == 1 else result["black_corners"]

            stats[winner_name]["wins"] += 1
            stats[loser_name]["losses"] += 1

            stats[winner_name]["win_discs"].append(winner_discs)
            stats[loser_name]["loss_discs"].append(loser_discs)
            stats[winner_name]["corner_wins"].append(winner_corners)

    # -----------------------------
    # Print Results
    # -----------------------------
    print("\n===== TOURNAMENT RESULTS =====\n")

    for player in stats:
        wins = stats[player]["wins"]
        losses = stats[player]["losses"]

        avg_win_discs = np.mean(stats[player]["win_discs"]) if stats[player]["win_discs"] else 0
        avg_loss_discs = np.mean(stats[player]["loss_discs"]) if stats[player]["loss_discs"] else 0
        avg_corner_wins = np.mean(stats[player]["corner_wins"]) if stats[player]["corner_wins"] else 0

        print(f"{player}:")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Avg Winning Discs: {avg_win_discs:.2f}")
        print(f"  Avg Losing Discs: {avg_loss_discs:.2f}")
        print(f"  Avg Corners in Wins: {avg_corner_wins:.2f}")
        print()

    print("===== PHASE AVERAGE CORNERS =====\n")

    for phase in ["early", "mid", "late"]:
        avg_white = phase_totals[phase]["white"] / num_games
        avg_black = phase_totals[phase]["black"] / num_games
        print(f"{phase.capitalize()} Phase:")
        print(f"  Avg White Corners: {avg_white:.2f}")
        print(f"  Avg Black Corners: {avg_black:.2f}")
        print()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python reversi_tournament.py player1 player2 num_games")
        sys.exit(1)

    player1_name = sys.argv[1]
    player2_name = sys.argv[2]
    num_games = int(sys.argv[3])

    run_tournament(player1_name, player2_name, num_games)

    # python reversi_tournament.py Better_Player greedy_player 100