#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi_ai import reversi

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

        
if __name__ == '__main__':
    main()