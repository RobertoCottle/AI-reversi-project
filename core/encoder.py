import numpy as np
import torch


def encode_board(board: np.ndarray, player: int) -> torch.Tensor:

    #Convert an 8x8 board array into a 6-plane tensor.
    # Tensors store the inputs, outputs, weights, and biases of a neural network

    #Planes:
    #    0: current player's pieces
    #    1: opponent's pieces
    #    2: empty squares
    #    3: corner positions (strategic landmarks)
    #    4: all 1s if current player is white, all 0s if black
    #    5: reserved (zeros)

    #Board convention from reversi.py:
    #    1  = white
    #   -1  = black
    #    0  = empty

    #Args:
    #    board:  8x8 numpy array from reversi.board
    #    player: 1 for white, -1 for black

    #Returns:
    #    torch.Tensor shape (6, 8, 8) float32
    
    planes = np.zeros((6, 8, 8), dtype=np.float32)

    planes[0] = (board == player).astype(np.float32)
    planes[1] = (board == -player).astype(np.float32)
    planes[2] = (board == 0).astype(np.float32)

    planes[3, 0, 0] = 1.0
    planes[3, 0, 7] = 1.0
    planes[3, 7, 0] = 1.0
    planes[3, 7, 7] = 1.0

    #if player == 1:
    #    planes[4] = np.ones((8, 8), dtype=np.float32)

    return torch.tensor(planes, dtype=torch.float32)


def get_legal_moves(board: np.ndarray, player: int) -> list:
    
    #Get all legal moves for player on board without using
    #a reversi game instance — pure numpy, no mutation risk.

    #Args:
    #    board:  8x8 numpy array
    #    player: 1 or -1

    #Returns:
    #    list of (x, y) tuples
    
    directions = [
        (1, 1), (1, 0), (1, -1), (0, 1),
        (0, -1), (-1, 1), (-1, 0), (-1, -1)
    ]
    legal = []

    for x in range(8):
        for y in range(8):
            if board[x, y] != 0:
                continue
            for dx, dy in directions:
                cx, cy = x + dx, y + dy
                found_opponent = False
                while 0 <= cx <= 7 and 0 <= cy <= 7:
                    if board[cx, cy] == 0:
                        break
                    elif board[cx, cy] == -player:
                        found_opponent = True
                        cx += dx
                        cy += dy
                    elif board[cx, cy] == player:
                        if found_opponent:
                            if (x, y) not in legal:
                                legal.append((x, y))
                        break
                    else:
                        break

    return legal


def apply_move(board: np.ndarray, x: int, y: int, player: int) -> np.ndarray:
    
    #Apply a move to a board and return the new board state.
    #Does not modify the input board.

    #Args:
    #    board:  8x8 numpy array
    #    x, y:   move coordinates
    #    player: 1 or -1

    #Returns:
    #    new 8x8 numpy array with move applied
    
    directions = [
        (1, 1), (1, 0), (1, -1), (0, 1),
        (0, -1), (-1, 1), (-1, 0), (-1, -1)
    ]
    new_board = board.copy()
    new_board[x, y] = player

    for dx, dy in directions:
        cx, cy = x + dx, y + dy
        flip_list = []
        while 0 <= cx <= 7 and 0 <= cy <= 7:
            if new_board[cx, cy] == 0:
                break
            elif new_board[cx, cy] == -player:
                flip_list.append((cx, cy))
                cx += dx
                cy += dy
            elif new_board[cx, cy] == player:
                for fx, fy in flip_list:
                    new_board[fx, fy] = player
                break
            else:
                break

    return new_board