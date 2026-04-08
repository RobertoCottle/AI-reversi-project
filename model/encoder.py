import numpy as np
import torch

def encode_board(board: np.ndarray, player: int) -> torch.Tensor:
    #Convert an 8x8 board array into a 6-plane tensor for the ResNet.

    #The six planes are:
    #    0: current player's pieces
    #    1: opponent's pieces
    #    2: empty squares
    #    3: corners (constant strategic reference)
    #    4: current player is white (all 1s or all 0s)
    #    5: move number normalized to [0, 1] — not used yet, zeroed

    #Args:
    #    board:  8x8 numpy array from reversi.board (1=white, -1=black)
    #    player: 1 for white, -1 for black

    #Returns:
    #    torch.Tensor of shape (6, 8, 8), dtype float32
    
    planes = np.zeros((6, 8, 8), dtype=np.float32)

    # Plane 0: current player's pieces
    planes[0] = (board == player).astype(np.float32)

    # Plane 1: opponent's pieces
    planes[1] = (board == -player).astype(np.float32)

    # Plane 2: empty squares
    planes[2] = (board == 0).astype(np.float32)

    # Plane 3: corners — constant strategic landmarks
    planes[3, 0, 0] = 1.0
    planes[3, 0, 7] = 1.0
    planes[3, 7, 0] = 1.0
    planes[3, 7, 7] = 1.0

    # Plane 4: which player is to move (all 1s = white, all 0s = black)
    if player == 1:
        planes[4] = np.ones((8, 8), dtype=np.float32)

    # Plane 5: reserved for move number, zeroed for now
    planes[5] = np.zeros((8, 8), dtype=np.float32)

    return torch.tensor(planes, dtype=torch.float32)