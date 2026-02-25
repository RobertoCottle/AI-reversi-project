# tournament_classroom.py
# Runs ## of games with Bot A as WHITE and ## games with Bot A as BLACK.
# Writes per-game results to tournament_results.csv.
# RECOMMENDATION: start with 2-5 games per color. This program isn't optimized to run super-fast. Can set higher. 

import csv
import importlib
import time
from reversi import reversi

# ---------------- CONFIG ----------------
BOT_A_MODULE = "Depth5_7"   # <-- change to whatever file (without .py)
BOT_B_MODULE = "Better_Player_random"   # <-- change to whatever file (without .py)

GAMES_PER_COLOR = 20
CSV_FILENAME = f"tournament_results_{int(time.time())}.csv"
PRINT_EACH_GAME = False
# ---------------------------------------


def load_bot(module_name: str):
    m = importlib.import_module(module_name)
    if not hasattr(m, "choose_move"):
        raise AttributeError(f"{module_name}.py must define choose_move(board, turn)")
    return m


BOT_A = load_bot(BOT_A_MODULE)
BOT_B = load_bot(BOT_B_MODULE)


def valid_moves(game: reversi, turn: int):
    moves = []
    for x in range(8):
        for y in range(8):
            if game.step(x, y, turn, commit=False) > 0:
                moves.append((x, y))
    return moves


def normalize_move(move):
    if move is None:
        return (-1, -1)
    return tuple(move)


def play_game(bot_white, bot_black):
    game = reversi()
    turn = game.turn  # classroom starts with WHITE (1)

    passes = 0
    moves_played = 0

    while True:
        moves = valid_moves(game, turn)

        if not moves:
            passes += 1
            if passes >= 2:
                break
            turn = -turn
            continue

        passes = 0

        if turn == 1:
            move = normalize_move(bot_white.choose_move(game.board, turn))
        else:
            move = normalize_move(bot_black.choose_move(game.board, turn))

        # pass (allowed)
        if move == (-1, -1):
            passes += 1
            if passes >= 2:
                break
            turn = -turn
            continue

        x, y = move
        flips = game.step(x, y, turn, commit=True)

        if flips <= 0:
            raise ValueError(
                f"Illegal move {move} for turn {turn}. Legal moves: {moves}"
            )

        moves_played += 1
        turn = -turn

    black_score = int(game.black_count)
    white_score = int(game.white_count)
    return black_score, white_score, moves_played


def main():
    print(f"Starting classroom-identical tournament...")
    print(f"Bot A = {BOT_A_MODULE}.py")
    print(f"Bot B = {BOT_B_MODULE}.py")

    rows = []
    game_id = 1

    a_wins = b_wins = draws = 0
    total_a_score = total_b_score = total_margin = 0

    def record(a_color, black_score, white_score, moves_played):
        nonlocal game_id, a_wins, b_wins, draws, total_a_score, total_b_score, total_margin, rows

        if a_color == "WHITE":
            a_score = white_score
            b_score = black_score
        else:
            a_score = black_score
            b_score = white_score

        if a_score > b_score:
            winner = "A"
            a_wins += 1
        elif b_score > a_score:
            winner = "B"
            b_wins += 1
        else:
            winner = "D"
            draws += 1

        total_a_score += a_score
        total_b_score += b_score
        total_margin += abs(a_score - b_score)

        rows.append({
            "game": game_id,
            "bot_a_color": a_color,
            "bot_b_color": "BLACK" if a_color == "WHITE" else "WHITE",
            "black_score": black_score,
            "white_score": white_score,
            "bot_a_score": a_score,
            "bot_b_score": b_score,
            "winner": winner,
            "margin_a_minus_b": a_score - b_score,
            "moves_played": moves_played,
        })

        if PRINT_EACH_GAME:
            print(
                f"Game {game_id:3d} | A({a_color}) {a_score:2d} - "
                f"B({'BLACK' if a_color == 'WHITE' else 'WHITE'}) {b_score:2d} | winner: {winner}"
            )

        game_id += 1

    # -------------------------
    # Block 1: Bot A as WHITE
    # -------------------------
    for _ in range(GAMES_PER_COLOR):
        b_score, w_score, moves_played = play_game(BOT_A, BOT_B)
        record("WHITE", b_score, w_score, moves_played)

    # -------------------------
    # Block 2: Bot A as BLACK
    # -------------------------
    for _ in range(GAMES_PER_COLOR):
        b_score, w_score, moves_played = play_game(BOT_B, BOT_A)
        record("BLACK", b_score, w_score, moves_played)

    total_games = 2 * GAMES_PER_COLOR

    with open(CSV_FILENAME, "w", newline="") as f:
        fieldnames = [
            "game",
            "bot_a_color",
            "bot_b_color",
            "black_score",
            "white_score",
            "bot_a_score",
            "bot_b_score",
            "winner",
            "margin_a_minus_b",
            "moves_played",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n==============================")
    print(f"FINAL RESULTS ({total_games} games)")
    print("==============================\n")
    print(f"Bot A Wins: {a_wins}/{total_games}")
    print(f"Bot B Wins: {b_wins}/{total_games}")
    print(f"Draws:      {draws}/{total_games}\n")
    print(f"Bot A avg score: {total_a_score / total_games:.2f}")
    print(f"Bot B avg score: {total_b_score / total_games:.2f}")
    print(f"Average margin:  {total_margin / total_games:.2f}")
    print(f"\nSaved per-game results to: {CSV_FILENAME}")


if __name__ == "__main__":
    main()