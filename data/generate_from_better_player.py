import os
import time
import numpy as np
import random
from reversi import reversi as reversi_game
from agents.alphabeta_agent import select_move_timed
from core.encoder import get_legal_moves, encode_board
from data.demo_buffer import DemoBuffer

# ── Import everything from the better player ──────────────────────────
from agents.better_player import (
    choose_move,
    clone_game_from_board,
    valid_moves,
    choose_depth,
    hasher,
    TT,
)

def alphabeta_move(board: np.ndarray, player: int) -> tuple:
    """AlphaBeta agent using negamax with 3s time limit."""
    return select_move_timed(board, player, time_limit=3.0, max_depth=3)

def greedy_move(board: np.ndarray, player: int) -> tuple:
    """Pick the move that flips the most discs immediately."""
    legal = get_legal_moves(board, player)
    if not legal:
        return -1, -1

    best_flips = -1
    best_x, best_y = legal[0]

    for lx, ly in legal:
        sim = reversi_game()
        sim.board = board.copy()
        flips = sim.step(lx, ly, player, commit=False)
        if flips > best_flips:
            best_flips = flips
            best_x, best_y = lx, ly

    return best_x, best_y

def play_one_game_better(
    white_func,
    black_func,
    random_opening_moves: int = 2,
) -> tuple:
    """
    Play one complete game.
    
    random_opening_moves: number of moves at the start where
    each agent has a 20% chance of playing a random legal move
    instead of their normal move. This breaks determinism and
    ensures different games even with the same agent pairing.
    """
    game    = reversi_game()
    history = []
    consecutive_passes = 0
    move_count = 0

    while True:
        player = game.turn
        g_tmp  = clone_game_from_board(game.board)
        legal  = valid_moves(g_tmp, player)

        if not legal:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.turn = -game.turn
            continue
        else:
            consecutive_passes = 0

        board_before = game.board.copy()

        # Select move
        if player == 1:
            agent_func = white_func
        else:
            agent_func = black_func

        # Random perturbation in opening to break determinism
        if move_count < random_opening_moves and np.random.random() < 0.3:
            x, y = legal[np.random.randint(len(legal))]
        else:
            x, y = agent_func(game.board, player)

        if x == -1 and y == -1:
            action = 64
            game.turn = -game.turn
            next_board = game.board.copy()
        else:
            action = x * 8 + y
            game.step(x, y, player, commit=True)
            game.turn = -game.turn
            next_board = game.board.copy()

        board_tensor      = encode_board(board_before, player).numpy()
        next_board_tensor = encode_board(next_board, -player).numpy()
        history.append((board_tensor, action, next_board_tensor, False))
        move_count += 1

    if history:
        b, a, nb, _ = history[-1]
        history[-1]  = (b, a, nb, True)

    white   = int(np.sum(game.board == 1))
    black   = int(np.sum(game.board == -1))
    outcome = 1.0 if white > black else -1.0 if black > white else 0.0

    return history, outcome


# def play_one_game_better(
#     white_func,
#     black_func,
# ) -> tuple:
#     """
#     Play one complete game between two agent functions.

#     Each function signature: func(board, player) -> (x, y)

#     Returns:
#         history: list of (board_tensor, action, next_board_tensor, done)
#         outcome: +1 white wins, -1 black wins, 0 draw
#     """
#     game    = reversi_game()
#     history = []
#     consecutive_passes = 0

#     while True:
#         player = game.turn
#         g_tmp  = clone_game_from_board(game.board)
#         legal  = valid_moves(g_tmp, player)

#         if not legal:
#             consecutive_passes += 1
#             if consecutive_passes >= 2:
#                 break
#             game.turn = -game.turn
#             continue
#         else:
#             consecutive_passes = 0

#         board_before = game.board.copy()

#         # Select move based on which color is playing
#         if player == 1:
#             x, y = white_func(game.board, player)
#         else:
#             x, y = black_func(game.board, player)

#         if x == -1 and y == -1:
#             action = 64
#             game.turn = -game.turn
#             next_board = game.board.copy()
#         else:
#             action = x * 8 + y
#             game.step(x, y, player, commit=True)
#             game.turn = -game.turn
#             next_board = game.board.copy()

#         board_tensor      = encode_board(board_before, player).numpy()
#         next_board_tensor = encode_board(next_board, -player).numpy()
#         history.append((board_tensor, action, next_board_tensor, False))

#     # Mark last transition as terminal
#     if history:
#         b, a, nb, _ = history[-1]
#         history[-1]  = (b, a, nb, True)

#     white   = int(np.sum(game.board == 1))
#     black   = int(np.sum(game.board == -1))
#     outcome = 1.0 if white > black else -1.0 if black > white else 0.0

#     return history, outcome


def random_move(board: np.ndarray, player: int) -> tuple:
    """Random legal move for diversity."""
    legal = get_legal_moves(board, player)
    if not legal:
        return -1, -1
    idx = np.random.randint(len(legal))
    return legal[idx]


def generate_expert_demos_better(
    save_path:  str   = "saved/demos/expert_demos_better.npz",
    n_games:    int   = 2000,
    log_every:  int   = 50,
    resume:     bool  = True,
    configs:    list  = None,
):
    """
    Generate expert demonstrations using the stronger BetterPlayer agent.

    Configs are (white_agent, black_agent, n_games) tuples where
    agent is one of 'better', 'random'.

    Using 'better' vs 'better' with color swapping ensures balanced
    win rates across white and black.
    """
    if configs is None:
        configs = [
            ('better', 'better', 1000),  # best vs best — half white, half black
            ('better', 'random',  500),  # better player as white
            ('random', 'better',  500),  # better player as black
        ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    buffer     = DemoBuffer(save_path)
    start_game = 0

    if resume and buffer.load():
        start_game = len(buffer) // 60
        print(f"[Generator] Resuming from ~{start_game} games")

    wins       = {'white': 0, 'black': 0, 'draw': 0}
    run_start  = time.time()
    games_done = 0
    total_games = sum(n for _, _, n in configs)

    print(f"[Generator] Generating {total_games} games using BetterPlayer")
    print(f"[Generator] Save every: {log_every} games")
    print("-" * 65)

    def get_agent_func(name: str):
        if name == 'better':
            return choose_move
        elif name == 'random':
            return random_move
        elif name == 'greedy':
            return greedy_move
        elif name == 'alphabeta':
            return alphabeta_move
        else:
            raise ValueError(f"Unknown agent: {name}")

    try:
        for white_name, black_name, n in configs:
            white_func = get_agent_func(white_name)
            black_func = get_agent_func(black_name)

            print(f"\n[Generator] {white_name}(W) vs {black_name}(B) "
                  f"— {n} games")

            config_wins = {'white': 0, 'black': 0, 'draw': 0}

            for game_idx in range(n):
                # Skip already-completed games when resuming
                if games_done < start_game:
                    games_done += 1
                    continue

                # Clear transposition table between games
                # to avoid stale values across games
                TT.clear()

                # For better vs better, randomly swap colors
                # every other game to ensure balance
                if white_name == 'better' and black_name == 'better':
                    if game_idx % 2 == 0:
                        wf, bf = white_func, black_func
                    else:
                        wf, bf = black_func, white_func
                else:
                    wf, bf = white_func, black_func

                history, outcome = play_one_game_better(wf, bf)

                for (board, action, next_board, done) in history:
                    reward = outcome if done else 0.0
                    buffer.push(board, action, reward, next_board, done)

                if outcome > 0:
                    wins['white']        += 1
                    config_wins['white'] += 1
                elif outcome < 0:
                    wins['black']        += 1
                    config_wins['black'] += 1
                else:
                    wins['draw']        += 1
                    config_wins['draw'] += 1

                games_done += 1

                if games_done % log_every == 0:
                    elapsed   = time.time() - run_start
                    rate      = games_done / elapsed if elapsed > 0 else 1
                    remaining = (total_games - games_done) / rate
                    total_w   = (wins['white'] + wins['black']
                                 + wins['draw'])

                    print(
                        f"  Game {games_done:>5}/{total_games} | "
                        f"W={wins['white']/max(total_w,1):.1%} "
                        f"B={wins['black']/max(total_w,1):.1%} "
                        f"D={wins['draw']/max(total_w,1):.1%} | "
                        f"trans={len(buffer):,} | "
                        f"elapsed={elapsed/60:.1f}min | "
                        f"ETA={remaining/60:.1f}min"
                    )
                    buffer.save()

            # Print config summary
            config_total = (config_wins['white'] + config_wins['black']
                           + config_wins['draw'])
            if config_total > 0:
                print(
                    f"  [{white_name} vs {black_name}] "
                    f"W={config_wins['white']/config_total:.1%} "
                    f"B={config_wins['black']/config_total:.1%}"
                )

    except KeyboardInterrupt:
        print(f"\n[Generator] Interrupted at game {games_done}")
        buffer.save()
        print(f"[Generator] Saved {len(buffer):,} transitions.")
        print(f"[Generator] Resume by running the script again.")
        return buffer

    buffer.save()
    total_time  = time.time() - run_start
    total_w     = wins['white'] + wins['black'] + wins['draw']

    print(f"\n[Generator] Complete — {games_done} games in "
          f"{total_time/60:.1f} minutes")
    print(f"[Generator] Total transitions: {len(buffer):,}")
    print(f"[Generator] Final balance: "
          f"W={wins['white']/max(total_w,1):.1%} "
          f"B={wins['black']/max(total_w,1):.1%} "
          f"D={wins['draw']/max(total_w,1):.1%}")
    return buffer