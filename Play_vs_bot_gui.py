import pygame
import importlib
from reversi import reversi

# ---------------- CONFIG ----------------
BOT_MODULE = "greedy_player"   # <- change to "greedy_player", "Good_Player_random", etc. (no .py)
HUMAN_COLOR = 1
WINDOW_SIZE = 640
MARGIN = 20
FPS = 60
# ---------------------------------------


def load_bot(name: str):
    m = importlib.import_module(name)
    if not hasattr(m, "choose_move"):
        raise AttributeError(f"{name}.py must define choose_move(board, turn)")
    return m


BOT = load_bot(BOT_MODULE)

GREEN = (25, 120, 70)
DARK = (10, 70, 40)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
HINT = (240, 220, 100)
TEXT = (240, 240, 240)
RED = (200, 60, 60)


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


def board_score(game: reversi):
    return int(game.black_count), int(game.white_count)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 80))
    pygame.display.set_caption(f"Reversi vs {BOT_MODULE} | You are {'WHITE' if HUMAN_COLOR == 1 else 'BLACK'}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)
    big = pygame.font.SysFont(None, 34)

    game = reversi()
    turn = game.turn  # classroom starts with WHITE (1)
    passes = 0
    game_over = False
    status = ""

    cell = (WINDOW_SIZE - 2 * MARGIN) // 8

    def to_pixel(x, y):
        px = MARGIN + y * cell
        py = MARGIN + x * cell
        return px, py

    def from_pixel(mx, my):
        if mx < MARGIN or my < MARGIN:
            return None
        y = (mx - MARGIN) // cell
        x = (my - MARGIN) // cell
        if 0 <= x < 8 and 0 <= y < 8:
            return int(x), int(y)
        return None

    def draw():
        screen.fill((0, 0, 0))
        # board background
        pygame.draw.rect(screen, GREEN, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
        pygame.draw.rect(screen, DARK, (MARGIN, MARGIN, cell * 8, cell * 8))

        for i in range(9):
            pygame.draw.line(screen, GREEN, (MARGIN, MARGIN + i * cell), (MARGIN + 8 * cell, MARGIN + i * cell), 2)
            pygame.draw.line(screen, GREEN, (MARGIN + i * cell, MARGIN), (MARGIN + i * cell, MARGIN + 8 * cell), 2)

        moves = valid_moves(game, turn) if (not game_over) else []
        move_set = set(moves)

        for (x, y) in move_set:
            px, py = to_pixel(x, y)
            pygame.draw.circle(screen, HINT, (px + cell // 2, py + cell // 2), cell // 10)

        # pieces
        for x in range(8):
            for y in range(8):
                v = int(game.board[x][y])
                if v == 0:
                    continue
                px, py = to_pixel(x, y)
                color = WHITE if v == 1 else BLACK
                pygame.draw.circle(screen, color, (px + cell // 2, py + cell // 2), cell // 2 - 6)

        # bottom panel
        pygame.draw.rect(screen, (25, 25, 25), (0, WINDOW_SIZE, WINDOW_SIZE, 80))
        b, w = board_score(game)
        turn_txt = "WHITE" if turn == 1 else "BLACK"
        you_txt = "WHITE" if HUMAN_COLOR == 1 else "BLACK"
        bot_txt = "BLACK" if HUMAN_COLOR == 1 else "WHITE"

        line1 = big.render(f"Score  BLACK: {b}   WHITE: {w}", True, TEXT)
        line2 = font.render(f"Turn: {turn_txt} | You: {you_txt} | Bot: {bot_txt} | {status}", True, TEXT)
        screen.blit(line1, (20, WINDOW_SIZE + 10))
        screen.blit(line2, (20, WINDOW_SIZE + 45))

        pygame.display.flip()

    def maybe_bot_move():
        nonlocal turn, passes, game_over, status
        if game_over:
            return
        if turn == HUMAN_COLOR:
            return

        moves = valid_moves(game, turn)
        if not moves:
            passes += 1
            status = "Bot passes."
            if passes >= 2:
                game_over = True
                status = "Game over."
            turn = -turn
            return

        passes = 0
        move = normalize_move(BOT.choose_move(game.board, turn))
        if move == (-1, -1):
            passes += 1
            status = "Bot passed (even though it had moves)."
            if passes >= 2:
                game_over = True
                status = "Game over."
            turn = -turn
            return

        flips = game.step(move[0], move[1], turn, commit=True)
        if flips <= 0:
            # bot made illegal move; end game with message
            game_over = True
            status = f"Bot played illegal move {move}. Game stopped."
            return

        status = f"Bot played {move}."
        turn = -turn

    running = True
    while running:
        clock.tick(FPS)

        # if it's bot's turn, it moves automatically (no delay)
        if not game_over and turn != HUMAN_COLOR:
            maybe_bot_move()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = reversi()
                    turn = game.turn
                    passes = 0
                    game_over = False
                    status = "Reset."

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_over:
                    continue
                if turn != HUMAN_COLOR:
                    continue

                pos = pygame.mouse.get_pos()
                sq = from_pixel(pos[0], pos[1])
                if sq is None:
                    continue

                moves = valid_moves(game, turn)
                if not moves:
                    passes += 1
                    status = "You pass."
                    if passes >= 2:
                        game_over = True
                        status = "Game over."
                    turn = -turn
                    continue

                if sq not in moves:
                    status = f"Illegal square {sq}. Click a highlighted move."
                    continue

                passes = 0
                game.step(sq[0], sq[1], turn, commit=True)
                status = f"You played {sq}."
                turn = -turn

        draw()

    pygame.quit()


if __name__ == "__main__":
    main()