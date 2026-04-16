import socket
import pickle
import time
import torch
import numpy as np
from core.encoder import get_legal_moves, encode_board
from network.q_network import QNetwork

HOST       = '127.0.0.1'
PORT       = 33333
MODEL_PATH = "saved/models/final_q.pt"
TIME_LIMIT = 5.0


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    print("[Competition] Connected to server")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Competition] Using device: {device}")

    model = QNetwork.load(MODEL_PATH, device)
    model.eval()
    print(f"[Competition] Model loaded from {MODEL_PATH}")

    color_identified = False

    while True:
        data       = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            print("[Competition] Game over")
            game_socket.close()
            return

        # Print color on first move received
        if not color_identified:
            color = "WHITE" if turn == 1 else "BLACK"
            color_identified = True
            print(f"[Competition] ─────────────────────────────")
            print(f"[Competition] Playing as: {color}")
            print(f"[Competition] ─────────────────────────────")

        legal = get_legal_moves(board, turn)

        if not legal:
            print("[Competition] No legal moves — passing")
            game_socket.send(pickle.dumps([-1, -1]))
            continue

        start = time.time()
        x, y  = model.predict_move(board, turn, legal, device)
        elapsed = time.time() - start

        print(f"[Competition] {'White' if turn==1 else 'Black'} "
              f"plays ({x},{y}) | time={elapsed:.3f}s")

        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()