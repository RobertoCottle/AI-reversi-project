import numpy as np
import socket
import pickle
from reversi import reversi
from core.encoder import get_legal_moves


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        game.board = board
        legal = get_legal_moves(board, turn)

        if legal:
            idx = np.random.randint(len(legal))
            x, y = legal[idx]
        else:
            x, y = -1, -1

        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()