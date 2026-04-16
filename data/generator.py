import os
import time
import numpy as np
import random
from reversi import reversi as reversi_game
from core.encoder import get_legal_moves, encode_board
from agents.alphabeta_agent import select_move_timed
from data.demo_buffer import DemoBuffer


def generate_expert_demos(
    save_path:      str   = "saved/demos/expert_demos.npz",
    n_games:        int   = 2000, #was 2000 before
    white_depth:    int   = 3,
    black_depth:    int   = 3,
    time_limit:     float = 1.0,
    log_every:      int   = 50,
    resume:         bool  = True,
):
    """
    Generate expert demonstration games using AlphaBeta agents.

    Supports Ctrl+C interruption and resume — saves progress
    every log_every games so you never lose more than that
    many games worth of data.

    Args:
        save_path:   where to save the demo buffer
        n_games:     total number of games to generate
        white_depth: AlphaBeta search depth for white
        black_depth: AlphaBeta search depth for black
        time_limit:  seconds per move
        log_every:   print progress and save every N games
        resume:      if True, load existing demos and continue
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    buffer     = DemoBuffer(save_path)
    start_game = 0

    if resume and buffer.load():
        # Estimate how many games are already done
        # Average game is ~60 transitions
        start_game = len(buffer) // 60
        print(f"[Generator] Resuming from ~{start_game} games")

    if start_game >= n_games:
        print(f"[Generator] Already have {start_game} games — nothing to do")
        return buffer

    wins   = {'white': 0, 'black': 0, 'draw': 0}
    run_start = time.time()

    print(f"[Generator] Generating {n_games - start_game} games "
          f"(AlphaBeta-d{white_depth} vs AlphaBeta-d{black_depth})")
    print(f"[Generator] Time limit: {time_limit}s per move")
    print(f"[Generator] Save every: {log_every} games")
    print("-" * 60)

    try:
        for game_idx in range(start_game, n_games):
            history, outcome = _play_one_game(
                white_depth, black_depth, time_limit
            )

            # Record transitions
            for (board, action, next_board, done) in history:
                if done:
                    # Terminal reward
                    reward = outcome
                else:
                    reward = 0.0
                buffer.push(board, action, reward, next_board, done)

            if outcome > 0:
                wins['white'] += 1
            elif outcome < 0:
                wins['black'] += 1
            else:
                wins['draw'] += 1

            games_done = game_idx - start_game + 1

            if games_done % log_every == 0:
                elapsed     = time.time() - run_start
                rate        = games_done / elapsed
                remaining   = (n_games - game_idx - 1) / rate if rate > 0 else 0
                total_so_far = game_idx + 1

                print(
                    f"  Game {total_so_far:>5}/{n_games} | "
                    f"W={wins['white']/games_done:.1%} "
                    f"B={wins['black']/games_done:.1%} "
                    f"D={wins['draw']/games_done:.1%} | "
                    f"trans={len(buffer):,} | "
                    f"elapsed={elapsed/60:.1f}min | "
                    f"ETA={remaining/60:.1f}min"
                )
                buffer.save()

    except KeyboardInterrupt:
        print(f"\n[Generator] Interrupted at game {game_idx + 1}")
        print(f"[Generator] Saving progress...")
        buffer.save()
        print(f"[Generator] Saved {len(buffer):,} transitions. "
              f"Resume by running the script again.")
        return buffer

    buffer.save()
    total_time = time.time() - run_start
    print(f"\n[Generator] Complete — {n_games} games in "
          f"{total_time/60:.1f} minutes")
    print(f"[Generator] Total transitions: {len(buffer):,}")
    return buffer


def _play_one_game(white_depth, black_depth, time_limit):
    # Randomly swap colors so training data is balanced
    if random.random() < 0.5:
        actual_white = white_depth
        actual_black = black_depth
    else:
        actual_white = black_depth
        actual_black = white_depth

    game    = reversi_game()
    history = []
    consecutive_passes = 0

    while True:
        player = game.turn
        legal  = get_legal_moves(game.board, player)

        if not legal:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            game.turn = -game.turn
            continue
        else:
            consecutive_passes = 0

        board_before = game.board.copy()
        depth = actual_white if player == 1 else actual_black

        x, y = select_move_timed(
            game.board, player,
            time_limit = time_limit,
            max_depth  = depth,
        )

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

    if history:
        b, a, nb, _ = history[-1]
        history[-1] = (b, a, nb, True)

    white   = int(np.sum(game.board == 1))
    black   = int(np.sum(game.board == -1))
    outcome = 1.0 if white > black else -1.0 if black > white else 0.0

    return history, outcome