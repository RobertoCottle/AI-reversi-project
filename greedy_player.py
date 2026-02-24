#Zijie Zhang, Sep.24/2023
#MAIN POINT OF EDITTING 
import numpy as np
import socket, pickle
from reversi import reversi

#specifically for tournament tester
def get_tourn_move(board, turn):
    game = reversi()
    game.board = board.copy()

    best_move = (-1,-1)
    max_flips = 0
    
    for i in range(8):
        for j in range(8):
            flips = game.step(i, j, turn, False) # simulate only

            if flips > max_flips:
                max_flips = flips
                best_move = (i, j)

    return best_move

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi() #connect through here for game board ---Donald 

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        print(turn, "Greedy_player.py")
        print(board)
        
        # --------------------EDITTABLE SECTION ------------------------------
        #Local Greedy - Replace with your algorithm 
        x = -1
        y = -1 
        max = 0 
        game.board = board
        for i in range(8):
            for j in range(8):
                cur = game.step(i, j, turn, False)
                if cur > max:
                    max = cur
                    x, y = i, j
                    ## develop algorthim for min maxxing with alpha-beta pruning
                    ## use evaluate heuristic from player perspective
                    ## use greedy ai to pick move that flips the most discs; tie-break randomly
        # --------------------END OF EDITTABLE SECTION ------------------------------
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y])) #this serializes the coordinates and sends the placement of the turn 
        
if __name__ == '__main__':
    main()