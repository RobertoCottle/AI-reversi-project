import json
import numpy as np
from reversi import reversi

#Opening lines
OPENING_LINES = """C4c3Diagonal Opening
C4c3D3c5B2X-square Opening
C4c3D3c5B3Snake/Peasant
C4c3D3c5B3f3Lysons
C4c3D3c5B3f4B5b4C6d6F5Pyramid/Checkerboarding Peasant
C4c3D3c5B4Heath/Tobidashi
C4c3D3c5B4d2C2f4D6c6F5e6F7Mimura Variation II
C4c3D3c5B4d2D6Heath-Bat
C4c3D3c5B4d2E2Iwasaki Variation
C4c3D3c5B4e3Heath-Chimney
C4c3D3c5B5Raccoon Dog
C4c3D3c5B6Rocket
C4c3D3c5B6c6B5Hamilton
C4c3D3c5B6e3Lollipop
C4c3D3c5D6Cow
C4c3D3c5D6e3Chimney
C4c3D3c5D6f4B4Cow Bat/Bat/Cambridge
C4c3D3c5D6f4B4b6B5c6B3Bat (Piau Continuation 2)
C4c3D3c5D6f4B4b6B5c6F5Melnikov/Bat (Piau Continuation 1)
C4c3D3c5D6f4B4c6B5b3B6e3C2a4A5a6D2Bat (Kling Continuation)
C4c3D3c5D6f4B4e3B3Bat (Kling Alternative)
C4c3D3c5D6f4F5Rose-v-Toth
C4c3D3c5D6f4F5d2Tanida
C4c3D3c5D6f4F5d2B5Aircraft/Feldborg
C4c3D3c5D6f4F5d2G4d7Sailboat
C4c3D3c5D6f4F5e6C6d7Maruoka
C4c3D3c5D6f4F5e6F6Landau
C4c3D3c5F6Buffalo/Kenichi Variation
C4c3D3c5F6e2C6Maruoka Buffalo
C4c3D3c5F6e3C6f5F4g5Tanida Buffalo
C4c3D3c5F6f5Hokuriku Buffalo
C4c3E6c5Wing Variation
C4c3F5c5Semi-Wing Variation
C4c5Parallel Opening
C4e3Perpendicular Opening
C4e3F4c5D6e6Mimura
C4e3F4c5D6f3C6Shaman/Danish
C4e3F4c5D6f3D3Inoue
C4e3F4c5D6f3D3c3Iago
C4e3F4c5D6f3E2Bhagat
C4e3F4c5D6f3E6c3D3e2Rose
C4e3F4c5D6f3E6c3D3e2B5Flat
C4e3F4c5D6f3E6c3D3e2B5f5Rotating Flat
C4e3F4c5D6f3E6c3D3e2B5f5B3Murakami Variation
C4e3F4c5D6f3E6c3D3e2B5f5B4f6C2e7D2c7Rotating Flat (Kling Continuation)
C4e3F4c5D6f3E6c3D3e2B6f5Rose-Birth
C4e3F4c5D6f3E6c3D3e2B6f5B4f6G5d7Brightstein
C4e3F4c5D6f3E6c3D3e2B6f5G5Rose-Birdie/Rose-Tamenori
C4e3F4c5D6f3E6c3D3e2B6f5G5f6Rose-Tamenori-Kling
C4e3F4c5D6f3E6c3D3e2D2Greenberg/Dawg
C4e3F4c5D6f3E6c6Ralle
C4e3F4c5E6Horse
C4e3F5b4Ganglion/No-Cat
C4e3F5b4F3Swallow
C4e3F5b4F3f4E2e6G5f6D6c6No-Cat (Continuation)
C4e3F5e6D3Italian
C4e3F5e6F4Cat
C4e3F5e6F4c5D6c6F7f3Sakaguchi
C4e3F5e6F4c5D6c6F7g5G6Berner
C4e3F6b4Bent Ganglion
C4e3F6e6F5Tiger
C4e3F6e6F5c5C3Stephenson
C4e3F6e6F5c5C3b4No-Kung
C4e3F6e6F5c5C3b4D6c6B5a6B6c7No-Kung (Continuation)
C4e3F6e6F5c5C3c6Comp'Oth
C4e3F6e6F5c5C3c6D3d2E2b3C1c2B4a3A5b5A6a4A2F.A.T. Draw
C4e3F6e6F5c5C3c6D6Lighning Bolt
C4e3F6e6F5c5C3g5Kung
C4e3F6e6F5c5D3Leader's Tiger
C4e3F6e6F5c5D6Brightwell
C4e3F6e6F5c5F4g5G4f3C6d3D6Ishii
C4e3F6e6F5c5F4g5G4f3C6d3D6b3C3b4E2b6Mainline Tiger
C4e3F6e6F5c5F4g6F7Rose-Bill
C4e3F6e6F5c5F4g6F7d3Tamenori
C4e3F6e6F5c5F4g6F7g5Central Rose-Bill/Dead Draw
C4e3F6e6F5g6Aubrey/Tanaka
C4e3F6e6F5g6E7c5Aubrey (Feldborg Continuation)"""

def parse_moves(line: str):
    #Extract move sequence from a line like 'C4c3D3c5B4Heath/Tobidashi'
    #Moves are letter+number pairs, uppercase or lowercase.
    #Returns list of (row, col) tuples in 0-indexed board coordinates.
    #
    moves = []
    i = 0
    while i < len(line) - 1:
        c = line[i]
        n = line[i + 1]
        # A move is a letter (a-h) followed by a digit (1-8)
        if c.isalpha() and n.isdigit():
            col = ord(c.lower()) - ord('a')  # a=0, b=1, ..., h=7
            row = int(n) - 1                  # 1=0, 2=1, ..., 8=7
            if 0 <= col < 8 and 0 <= row < 8:
                moves.append((row, col))
            i += 2
        else:
            break  # hit the name portion, stop
    return moves

def board_to_key(board: np.ndarray, player: int) -> str:
    # Cast to float64 to match reversi class default dtype
    return board.astype(np.float64).tobytes().hex() + str(player)

def init_board() -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.float64)  # match np.zeros([8,8]) default float
    board[3, 4] = -1  # black
    board[3, 3] = 1   # white
    board[4, 3] = -1  # black
    board[4, 4] = 1   # white
    return board

def build_opening_book():
    book = {}
    skipped = 0

    for line in OPENING_LINES.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        moves = parse_moves(line)
        if not moves:
            skipped += 1
            continue

        # Replay moves from start, recording each board state -> next move
        board = init_board()
        g = reversi()
        g.board = board

        # Determine starting player
        # Black (-1) always moves first in standard Reversi
        turn = 1

        for idx, (row, col) in enumerate(moves):
            key = board_to_key(g.board, turn)

            # Only store if not already in book
            # First entry wins — earlier/more common lines take priority
            if key not in book:
                book[key] = [row, col]

            # Apply move
            flips = g.step(row, col, turn, commit=True)
            if flips == 0:
                # Invalid move in this line — stop processing this opening
                break

            turn = -turn  # alternate turns

    print(f"[BOOK] Built {len(book)} positions, skipped {skipped} unparseable lines")
    return book


def save_book(book: dict, path: str = "opening_book.json"):
    with open(path, "w") as f:
        json.dump(book, f, separators=(',', ':'))  # compact, no whitespace = faster load
    print(f"[BOOK] Saved to {path}")

if __name__ == "__main__":
    book = build_opening_book()
    save_book(book)

