#Zijie Zhang, Sep.24/2023
#FOCUS IS TO CHANGE THE ALGORITHM
import numpy as np
import socket, pickle
from reversi import reversi

def minimax(game, depth, alpha, beta, current_turn, player):
    if depth == 0:
        return evaluate(game, player)

    moves = get_valid_moves(game, current_turn)

    if not moves:
        return evaluate(game, player)

    if current_turn == player:  # MAXIMIZING
        max_eval = -float('inf')
        for move in moves:
            new_game = copy_game(game)
            new_game.step(move[0], move[1], current_turn)
            eval = minimax(new_game, depth-1, alpha, beta, -current_turn, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # PRUNE
        return max_eval

    else:  # MINIMIZING
        min_eval = float('inf')
        for move in moves:
            new_game = copy_game(game)
            new_game.step(move[0], move[1], current_turn)
            eval = minimax(new_game, depth-1, alpha, beta, -current_turn, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # PRUNE
        return min_eval
    
# Helper for returning all possible valid positions
def get_valid_moves(game, turn):
    moves = []
    for i in range(8):
        for j in range(8):
            if game.step(i, j, turn, False) > 0:
                moves.append((i, j))
    return moves

# evaluation function for minmax
def evaluate(game, player):
    return player * (game.white_count - game.black_count)

# Copy game logic, to not modify the real board
def copy_game(game):
    new_game = reversi()
    new_game.board = np.copy(game.board)
    new_game.white_count = game.white_count
    new_game.black_count = game.black_count
    return new_game

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi() #connect through here ---Donald 

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
        print(turn, "min-maxplayer")
        print(board)
        
        # --------------------EDITTABLE SECTION (DO NOT EDIT THE REST OF MAIN------------------------
        #Local Greedy - Replace with your algorithm 
        # x = -1
        # y = -1 
        # max = 0 
        # game.board = board
        # for i in range(8):
        #     for j in range(8):
        #         cur = game.step(i, j, turn, False)
        #         if cur > max:
        #             max = cur
        #             x, y = i, j
        depth = 6 #was 3 before
        best_score = -float('inf')
        best_move = (-1,-1)

        game.board = board
        moves = get_valid_moves(game, turn)

        for move in moves:
            new_game = copy_game(game)
            new_game.step(move[0], move[1], turn)

            score = minimax(new_game, depth-1, -float('inf'), float('inf'), -turn, turn)

            if score > best_score:
                best_score = score
                best_move = move

        x, y = best_move
        # --------------------END OF EDITTABLE SECTION ------------------------------
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y])) #this serializes the coordinates and sends the placement of the turn 
        
if __name__ == '__main__':
    main()

# minmax function
# def minimax(game, depth, maximizing_player, player):
#     if depth == 0:
#         return evaluate(game, player)

#     moves = get_valid_moves(game, maximizing_player)

#     if not moves:
#         return evaluate(game, player)

#     if maximizing_player == player:
#         max_eval = -float('inf')
#         for move in moves:
#             new_game = copy_game(game)
#             new_game.step(move[0], move[1], maximizing_player)
#             eval = minimax(new_game, depth-1, -maximizing_player, player)
#             max_eval = max(max_eval, eval)
#         return max_eval
#     else:
#         min_eval = float('inf')
#         for move in moves:
#             new_game = copy_game(game)
#             new_game.step(move[0], move[1], maximizing_player)
#             eval = minimax(new_game, depth-1, -maximizing_player, player)
#             min_eval = min(min_eval, eval)
#         return min_eval