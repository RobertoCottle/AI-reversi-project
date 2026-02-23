import numpy as np
import socket, pickle
from reversi import reversi
import random


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        # Receive play request from the server
        # turn : 1 --> you are playing as white | -1 --> you are playing as black
        # board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        # Debug info
        print(turn)
        print(board)

        # Random "best move" Greedy
        x = -1
        y = -1
        max_val = 0
        best_moves = []

        game.board = board

        for i in range(8):
            for j in range(8):
                cur = game.step(i, j, turn, False)

                if cur > max_val:
                    max_val = cur
                    best_moves = [(i, j)]
                elif cur == max_val and cur > 0:
                    best_moves.append((i, j))

        if best_moves:
            x, y = random.choice(best_moves)

        # Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()