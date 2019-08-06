#! /usr/bin/env python

'''
strategies: minimax, negascout, negamax, minimax w/ab cutoff
compile: python setup.py py2exe (+ add font and background)
'''

from copy import deepcopy # http://www.wellho.net/resources/ex.php4?item=y111/deepcop.py
import random # http://effbot.org/pyfaq/how-do-i-generate-random-numbers-in-python.htm
import numpy as np
import sys # import exit function
# gui imports
import pygame # import pygame package
from pygame.locals import * # import values and constants

from numpy.distutils.fcompiler import none
from numpy.lib.function_base import delete

import NN
from matplotlib._layoutbox import hpack
######################## VARIABLES ########################

NN.turn=0
turn = 'black' # keep track of whose turn it is
selected = (0, 1) # a tuple keeping track of which piece is selected
board = 0 # link to our 'main' board
move_limit = [160, 0] # move limit for each game (declares game as draw otherwise)

# artificial intelligence related
best_move = () # best move for the player as determined by strategy
black, white = (), () # black and white players

# gui variables
window_size = (256, 256) # size of board in pixels
background_image_filename = 'board_brown.png' # image for the background
title = 'pyCheckers 1.1.2.3 final' # window title
board_size = 8 # board is 8x8 squares
left = 1 # left mouse button
fps = 5 # framerate of the scene (to save cpu time)
pause = 1 # number of seconds to pause the game for after end of game
start = True # are we at the beginnig of the game?
model=NN.create_model()
model2=NN.create_model()
model.load_weights("./training_1/cp.mdl_wts.hdf5")
model2.load_weights("./training_2/cp.mdl_wts.hdf5")
time=0
######################## CLASSES ########################

# class representing piece on the board
class Piece(object):
    def __init__(self, color, king):
        self.color = color
        self.king = king

# class representing player
class Player(object):
    def __init__(self, type, color, strategy, ply_depth):
        self.type = type # cpu or human
        self.color = color # black or white
        self.strategy = strategy # choice of strategy: minimax, negascout, negamax, minimax w/ab
        self.ply_depth = ply_depth # ply depth for algorithms

######################## INITIALIZE ########################

# will initialize board with all the pieces
def init_board():
    global move_limit
    move_limit[1] = 0 # reset move limit
    result = [
    [ 0, 1, 0, 1, 0, 1, 0, 1],
    [ 1, 0, 1, 0, 1, 0, 1, 0],
    [ 0, 1, 0, 1, 0, 1, 0, 1],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0,-1, 0,-1, 0,-1, 0],
    [ 0,-1, 0,-1, 0,-1, 0,-1],
    [-1, 0,-1, 0,-1, 0,-1, 0]
    ] # initial board setting
    for m in range(8):
        for n in range(8):
            if (result[m][n] == 1):
                piece = Piece('black', False) # basic black piece
                result[m][n] = piece
            elif (result[m][n] == -1):
                piece = Piece('white', False) # basic white piece
                result[m][n] = piece
    return result

# initialize players
def init_player(type, color, strategy, ply_depth):
    return Player(type, color, strategy, ply_depth)

######################## FUNCTIONS ########################

# will return array with available moves to the player on board
def avail_moves(board, player):
    moves = np.empty([0,4],int) # will store available jumps and moves
    global jump
    for m in range(8):
        for n in range(8):
            if board[m][n] != 0 and board[m][n].color == player: # for all the players pieces...
                # ...check for jumps first
                aux_board = deepcopy(board)
                aux = check_four(m,n,aux_board)
                
                if len(aux) > 0:
                     moves=np.append(moves,aux,axis=0)
                if(len(aux)>1):
                    aux=aux
                jump=True
    if len(moves) == 0: # if there are no jumps in the list (no jumps available)
        # ...check for regular moves
        jump=False
        for m in range(8):
            for n in range(8):
                if board[m][n] != 0 and board[m][n].color == player: # for all the players pieces...
                    if can_move([m, n], [m+1, n+1], board) == True: moves=np.append(moves,[[m, n, m+1, n+1]],axis=0)
                    if can_move([m, n], [m-1, n+1], board) == True: moves=np.append(moves,[[m, n, m-1, n+1]],axis=0)
                    if can_move([m, n], [m+1, n-1], board) == True: moves=np.append(moves,[[m, n, m+1, n-1]],axis=0)
                    if can_move([m, n], [m-1, n-1], board) == True: moves=np.append(moves,[[m, n, m-1, n-1]],axis=0)
                    
    return moves # return the list with available jumps or moves

def check_four(m, n,board):
    stage_board = deepcopy(board)
    listMove=np.zeros((0,4),int)
    
    if can_jump([m, n], [m+1, n+1], [m+2, n+2], board):
        listMove= np.row_stack((listMove,[m, n, m+2, n+2]))
        listMove= np.row_stack((listMove,check_four(m+2,n+2,board)))
        board= deepcopy(stage_board)
    if can_jump([m, n], [m-1, n+1], [m-2, n+2], board):
        listMove= np.row_stack((listMove,[m, n, m-2, n+2]))
        listMove= np.row_stack((listMove,check_four(m-2,n+2,board)))
        board= deepcopy(stage_board)
    if can_jump([m, n], [m+1, n-1], [m+2, n-2], board):
        listMove= np.row_stack((listMove,[m, n, m+2, n-2]))
        listMove= np.row_stack((listMove,check_four(m+2,n-2,board)))
        board= deepcopy(stage_board)
    if can_jump([m, n], [m-1, n-1], [m-2, n-2], board):
        listMove= np.row_stack((listMove,[m, n, m-2, n-2]))
        listMove= np.row_stack((listMove,check_four(m-2,n-2,board)))
        board= deepcopy(stage_board)
    return listMove
                
# will return true if the jump is legal
def can_jump(a, via, b, board):
    # is destination off board?
    if b[0] < 0 or b[0] > 7 or b[1] < 0 or b[1] > 7:
        return False
    # does destination contain a piece already?
    if board[b[0]][b[1]] != 0: return False
    # are we jumping something?
    if board[via[0]][via[1]] == 0: return False
    # for white piece
    if board[a[0]][a[1]].color == 'white':
        if board[a[0]][a[1]].king == False and b[0] > a[0]: return False # only move up
        if board[via[0]][via[1]].color != 'black': return False # only jump blacks
        make_move([np.concatenate((a,b))],0,board)
        return True # jump is possible
    # for black piece
    if board[a[0]][a[1]].color == 'black':
        if board[a[0]][a[1]].king == False and b[0] < a[0]: return False # only move down
        if board[via[0]][via[1]].color != 'white': return False # only jump whites
        make_move([np.concatenate((a,b))],0,board)
        return True # jump is possible

# will return true if the move is legal
def can_move(a, b, board):
    # is destination off board?
    if b[0] < 0 or b[0] > 7 or b[1] < 0 or b[1] > 7:
        return False
    # does destination contain a piece already?
    if board[b[0]][b[1]] != 0: return False
    # for white piece (not king)
    if board[a[0]][a[1]].king == False and board[a[0]][a[1]].color == 'white':
        if b[0] > a[0]: return False # only move up
        return True # move is possible
    # for black piece
    if board[a[0]][a[1]].king == False and board[a[0]][a[1]].color == 'black':
        if b[0] < a[0]: return False # only move down
        return True # move is possible
    # for kings
    if board[a[0]][a[1]].king == True: return True # move is possible

# make a move on a board, assuming it's legit
def make_move(moves,i, board):
    try:
        if type(board[moves[i][0]][moves[i][1]])==int:
            for m in range(len(moves)):
                if (moves[m][2],moves[m][3]) == (moves[i][0],moves[i][1]):
                    make_move(moves,m,board)
                    break
            board[moves[i][2]][moves[i][3]] = board[moves[i][0]][moves[i][1]] # make the move
            board[moves[i][0]][moves[i][1]] = 0
        else:
            board[moves[i][2]][moves[i][3]] = board[moves[i][0]][moves[i][1]] # make the move
            board[moves[i][0]][moves[i][1]] = 0 # delete the source
                
            # check if we made a king - there is an error here that can not identify - its ok just ignore
        try:
            if moves[i][2] == 0 and board[moves[i][2]][moves[i][3]].color == 'white': board[moves[i][2]][moves[i][3]].king = True
            if moves[i][2] == 7 and board[moves[i][2]][moves[i][3]].color == 'black': board[moves[i][2]][moves[i][3]].king = True
        except:
            a=0
        if (moves[i][0] - moves[i][2]) % 2 == 0: # we made a jump...
            board[int((moves[i][0]+moves[i][2])/2)][int((moves[i][1]+moves[i][3])/2)] = 0 # delete the jumped piece
    except:
        a=0

######################## CORE FUNCTIONS ########################

# will evaluate board for a player
def evaluate(game, player):

    ''' this function just adds up the pieces on board (100 = piece, 175 = king) and returns the difference '''
    def simple_score(game, player):
        black, white = 0, 0 # keep track of score
        for m in range(8):
            for n in range(8):
                if (game[m][n] != 0 and game[m][n].color == 'black'): # select black pieces on board
                    if game[m][n].king == False: black += 100 # 100pt for normal pieces
                    else: black += 175 # 175pts for kings
                elif (game[m][n] != 0 and game[m][n].color == 'white'): # select white pieces on board
                    if game[m][n].king == False: white += 100 # 100pt for normal pieces
                    else: white += 175 # 175pts for kings
        if player != 'black': return white-black
        else: return black-white

    ''' this function will add bonus to pieces going to opposing side '''
    def piece_rank(game, player):
        black, white = 0, 0 # keep track of score
        for m in range(8):
            for n in range(8):
                if (game[m][n] != 0 and game[m][n].color == 'black'): # select black pieces on board
                    if game[m][n].king != True: # not for kings
                        black = black #+ (m*m)
                elif (game[m][n] != 0 and game[m][n].color == 'white'): # select white pieces on board
                    if game[m][n].king != True: # not for kings
                        white = white  #+ ((7-m)*(7-m))
        if player != 'black': return (white-black)
        else: return black-white

    ''' a king on an edge could become trapped, thus deduce some points '''
    def edge_king(game, player):
        black, white = 0, 0 # keep track of score
        for m in range(8):
            if (game[m][0] != 0 and game[m][0].king != False):
                if game[m][0].color != 'white': black += -25
                else: white += -25
            if (game[m][7] != 0 and game[m][7].king != False):
                if game[m][7].color != 'white': black += -25
                else: white += -25
        if player != 'black': return white-black
        else: return black-white
    
    multi = random.uniform(0.90, 1.10) # will add +/- 10 percent to the score to make things more unpredictable

    return (simple_score(game, player) + piece_rank(game, player) + edge_king(game, player)) * multi

# have we killed the opponent already?
def end_game(board):
    black, white = 0, 0 # keep track of score
    for m in range(8):
        for n in range(8):
            if board[m][n] != 0:
                if board[m][n].color == 'black': black += 1 # we see a black piece
                else: white += 1 # we see a white piece

    return black, white

def neural_check_four(a,board):
    move=deepcopy(a)
    move=np.reshape(move,(4))
    if can_jump((move[2],move[3]), (move[2]+1,move[3]+1), (move[2]+2,move[3]+2), board) or can_jump((move[2],move[3]), (move[2]+1,move[3]-1), (move[2]+2,move[3]-2), board) or can_jump((move[2],move[3]), (move[2]-1,move[3]+1), (move[2]-2,move[3]+2), board) or can_jump((move[2],move[3]), (move[2]-1,move[3]-1), (move[2]-2,move[3]-2), board):
        return True
    return False

def avail_neural_moves(board,player):
    global best_move, move_limit
    
    if NN.turn<5:
        moves=NN.predict(board, avail_moves(board, player),model,1)
    elif NN.turn<14:
        moves=NN.predict(board, avail_moves(board, player),model,3)
    elif NN.turn < 24:
        moves=NN.predict(board, avail_moves(board, player),model2,4)
    else:
        moves=NN.predict(board, avail_moves(board, player),model2,6)
    
    return moves

def neural_max(board, player, ply):
    global best_move

    end = end_game(board)
    if player.color == 'black': enemy = 'white'
    else: enemy = 'black'
    
    ''' if node is a terminal node or depth = CutoffDepth '''
    if ply >= player.ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
        ''' return the heuristic value of node '''
        return evaluate(board, player.color)

    ''' if the adversary is to play at node '''
    if ply%2!=0: # if the opponent is to play on this node...
        
        ''' let beta := +infinity '''
        beta = +10000
        
        
        ''' foreach child of node '''
        moves = avail_neural_moves(board, enemy) # get the available moves for player
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves,i, new_board) # make move on new board
                                
            temp_beta = neural_max(new_board, player, ply+1)-i
            if temp_beta < beta:
                beta = temp_beta # take the lowest beta

        ''' return beta '''
        return beta
    
    else: # else we are to play
        ''' else {we are to play at node} '''
        ''' let alpha := -infinity '''
        alpha = -10000
        
        ''' foreach child of node '''
        moves = avail_neural_moves(board, player.color) # get the available moves for player
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves, i, new_board) # make move on new board
            
            temp_alpha = neural_max(new_board, player, ply+1)-i
            if temp_alpha > alpha:
                alpha = temp_alpha # take the highest alpha
                if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move as it's our turn

        ''' return alpha '''
        return alpha
    

# will generate possible moves and board states until a given depth
''' http://en.wikipedia.org/wiki/Minimax '''
''' function minimax(node, depth) '''
def minimax(board, player, ply):
    global best_move

    end = end_game(board)
    if player.color == 'black': enemy = 'white'
    else: enemy = 'black'
    
    ''' if node is a terminal node or depth = CutoffDepth '''
    if ply >= player.ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
        ''' return the heuristic value of node '''
        return evaluate(board, player.color)

    ''' if the adversary is to play at node '''
    if ply%2!=0: # if the opponent is to play on this node...
        
        ''' let beta := +infinity '''
        beta = +10000
        
        
        ''' foreach child of node '''
        moves = avail_moves(board, enemy) # get the available moves for player
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves,i, new_board) # make move on new board
                                
            temp_beta = minimax(new_board, player, ply+1)
            if temp_beta < beta:
                beta = temp_beta # take the lowest beta

        ''' return beta '''
        return beta
    
    else: # else we are to play
        ''' else {we are to play at node} '''
        ''' let alpha := -infinity '''
        alpha = -10000
        
        ''' foreach child of node '''
        moves = avail_moves(board, player.color) # get the available moves for player
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves, i, new_board) # make move on new board
            
            temp_alpha = minimax(new_board, player, ply+1)
            if temp_alpha > alpha:
                alpha = temp_alpha # take the highest alpha
                if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move as it's our turn

        ''' return alpha '''
        return alpha

''' http://en.wikipedia.org/wiki/Negascout '''
''' function negascout(node, depth, alpha, beta) '''
def negascout(board, ply, alpha, beta, player):
    global best_move

    # find out ply depth for player
    ply_depth = 0
    if player != 'black': ply_depth = white.ply_depth
    else: ply_depth = black.ply_depth

    end = end_game(board)
    
    ''' if node is a terminal node or depth = 0 '''
    if ply >= ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
        ''' return the heuristic value of node '''
        score = evaluate(board, player) # return evaluation of board as we have reached final ply or end state
        return score
    ''' b := beta '''
    b = beta

    ''' foreach child of node '''
    moves = avail_moves(board, player) # get the available moves for player
    for i in range(len(moves)):
        # create a deep copy of the board (otherwise pieces would be just references)
        new_board = deepcopy(board)
        make_move(moves,i, new_board) # make move on new board

        ''' alpha := -negascout (child, depth-1, -b, -alpha) '''
        # ...make a switch of players
        if player == 'black': player = 'white'
        else: player = 'black'

        alpha = -negascout(new_board, ply+1, -b, -alpha, player)
        ''' if alpha >= beta '''
        if alpha >= beta:
            ''' return alpha '''
            return alpha # beta cut-off
        ''' if alpha >= b '''
        if alpha >= b: # check if null-window failed high

            ''' alpha := -negascout(child, depth-1, -beta, -alpha) '''
            # ...make a switch of players
            if player == 'black': player = 'white'
            else: player = 'black'

            alpha = -negascout(new_board, ply+1, -beta, -alpha, player) # full re-search
            ''' if alpha >= beta '''
            if alpha >= beta:
                ''' return alpha '''
                return alpha # beta cut-off
        ''' b := alpha+1 '''
        b = alpha+1 # set new null window
    ''' return alpha '''
    if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
    return alpha

''' http://en.wikipedia.org/wiki/Negamax '''
''' function negamax(node, depth, alpha, beta) '''
def negamax(board, ply, alpha, beta, player):
    global best_move
    
    # find out ply depth for player
    ply_depth = 0
    if player != 'black': ply_depth = white.ply_depth
    else: ply_depth = black.ply_depth

    end = end_game(board)

    ''' if node is a terminal node or depth = 0 '''
    if ply >= ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
        ''' return the heuristic value of node '''
        score = evaluate(board, player) # return evaluation of board as we have reached final ply or end state
        return score

    ''' else '''
    ''' foreach child of node '''
    moves = avail_moves(board, player) # get the available moves for player
    for i in range(len(moves)):
        # create a deep copy of the board (otherwise pieces would be just references)
        new_board = deepcopy(board)
        make_move(moves,i, new_board) # make move on new board

        ''' alpha := max(alpha, -negamax(child, depth-1, -beta, -alpha)) '''
        # ...make a switch of players
        if player == 'black': player = 'white'
        else: player = 'black'

        temp_alpha = -negamax(new_board, ply+1, -beta, -alpha, player)
        if temp_alpha >= alpha:
            if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
            alpha = temp_alpha

        ''' {the following if statement constitutes alpha-beta pruning} '''
        ''' if alpha>=beta '''
        if alpha >= beta:
            ''' return beta '''
            if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
            return beta
    ''' return alpha '''
    return alpha

''' http://www.ocf.berkeley.edu/~yosenl/extras/alphabeta/alphabeta.html '''
''' alpha-beta(player,board,alpha,beta) '''
def alpha_beta(player, board, ply, alpha, beta):
    global best_move

    # find out ply depth for player
    ply_depth = 0
    if player != 'black': ply_depth = white.ply_depth
    else: ply_depth = black.ply_depth

    end = end_game(board)

    ''' if(game over in current board position) '''
    if ply >= ply_depth or end[0] == 0 or end[1] == 0: # are we still playing?
        ''' return winner '''
        score = evaluate(board, player) # return evaluation of board as we have reached final ply or end state
        return score

    ''' children = all legal moves for player from this board '''
    moves = avail_moves(board, player) # get the available moves for player

    ''' if(max's turn) '''
    if player == turn: # if we are to play on node...
        ''' for each child '''
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves,i, new_board) # make move on new board

            ''' score = alpha-beta(other player,child,alpha,beta) '''
            # ...make a switch of players for minimax...
            if player == 'black': player = 'white'
            else: player = 'black'

            score = alpha_beta(player, new_board, ply+1, alpha, beta)

            ''' if score > alpha then alpha = score (we have found a better best move) '''
            if score > alpha:
                if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
                alpha = score
            ''' if alpha >= beta then return alpha (cut off) '''
            if alpha >= beta:
                #if ply == 0: best_move = (moves[i][0], moves[i][1]), (moves[i][2], moves[i][3]) # save the move
                return alpha

        ''' return alpha (this is our best move) '''
        return alpha

    else: # the opponent is to play on this node...
        ''' else (min's turn) '''
        ''' for each child '''
        for i in range(len(moves)):
            # create a deep copy of the board (otherwise pieces would be just references)
            new_board = deepcopy(board)
            make_move(moves,i, new_board) # make move on new board

            ''' score = alpha-beta(other player,child,alpha,beta) '''
            # ...make a switch of players for minimax...
            if player == 'black': player = 'white'
            else: player = 'black'

            score = alpha_beta(player, new_board, ply+1, alpha, beta)

            ''' if score < beta then beta = score (opponent has found a better worse move) '''
            if score < beta: beta = score
            ''' if alpha >= beta then return beta (cut off) '''
            if alpha >= beta: return beta
        ''' return beta (this is the opponent's best move) '''
        return beta

# end turn
def end_turn():
    global turn # use global variables
    
    if turn != 'black':    turn = 'black'
    else: turn = 'white'

# play as a computer
def cpu_play(player):
    global board, move_limit, jump # global variables
    
    # find and print the best move for cpu
    if player.strategy == 'minimax': 
        alpha = minimax(board, player, 0)
        #a=board[534545]
    elif player.strategy == 'negascout': alpha = negascout(board, 0, -10000, +10000, player.color)
    elif player.strategy == 'negamax': alpha = negamax(board, 0, -10000, +10000, player.color)
    elif player.strategy == 'alpha-beta': alpha = alpha_beta(player.color, board, 0, -10000, +10000)
    elif player.strategy == 'neural_max': 
        alpha = neural_max(board, player,0)
        NN.turn=NN.turn+1
    #print player.color, alpha

    if alpha == -10000: # no more moves available... all is lost
        if player.color == white: show_winner("black")
        else: show_winner("white")
    
    moves = avail_moves(board, turn)
    try:
        make_sequential_move(moves,best_move, board)
    except:
        make_sequential_move(moves,best_move, board)
    if player.strategy == 'neural_max' and neural_check_four(best_move, deepcopy(board)):
        if jump:
            return
    end_turn() # end turn

def make_sequential_move(moves,move,board):
    if type(board[move[0][0]][move[0][1]])==int:
            for m in range(len(moves)):
                if (moves[m][2],moves[m][3])==(move[0][0],move[0][1]):
                    make_sequential_move(moves,np.reshape(moves[m],(2,2)),board)
                    
    make_move([np.concatenate((move[0],move[1]))],0, board) # make the move on board
    move_limit[1] += 1 # add to move limit
    
# make changes to ply's if playing vs human (problem with scope)
def ply_check():
    global black, white

    ''' if human has higher ply_setting, cpu will do unnecessary calculations '''
    if black.type != 'cpu': black.ply_depth = white.ply_depth
    elif white.type != 'cpu': white.ply_depth = black.ply_depth

# will check for errors in players settings
def player_check():
    global black, white

    if black.type != 'cpu' or black.type != 'human': black.type = 'cpu'
    if white.type != 'cpu' or white.type != 'human': white.type = 'cpu'

    if black.ply_depth <0: black.ply_depth = 1
    if white.ply_depth <0: white.ply_depth = 1

    if black.color != 'black': black.color = 'black'
    if white.color != 'white': white.color = 'white'

    if black.strategy != 'minimax' or black.strategy != 'negascout' or black.strategy!= 'neural_max':
        if black.strategy != 'negamax' or black.strategy != 'alpha-beta': black.strategy = 'alpha-beta'
    if white.strategy != 'minimax' or white.strategy != 'negascout' or white.strategy!= 'neural_max':
        if white.strategy != 'negamax' or white.strategy != 'alpha-beta': white.strategy = 'alpha-beta'

# initialize players and the boardfor the game
def game_init(difficulty=""):
    global black, white # work with global variables
    # init
    if difficulty == "":
        black = init_player('cpu', 'black', 'neural_max', 4) # init black player
        white = init_player('cpu', 'white', 'minimax', 4) # init white player
        board = init_board()
    # reset
    elif difficulty == " ":
        board = init_board()
    # white as human
    elif difficulty == "human":
        white = init_player('human', 'white', 'minimax', white.ply_depth)
        board = init_board()
    # white as minimax
    else:
        white = init_player('cpu', 'white', 'minimax', white.ply_depth) # init white player
        board = init_board()
    
    NN.black=0
    NN.white=0
    NN.draw=0
    return board            

def printResult():
    print (NN.black,"-", NN.draw,"-",NN.white)
######################## GUI FUNCTIONS ########################

# function that will draw a piece on the board
def draw_piece(row, column, color, king):
    # find the center pixel for the piece
    posX = int(((window_size[0]/8)*column) - (window_size[0]/8)/2)
    posY = int(((window_size[1]/8)*row) - (window_size[1]/8)/2)
    
    # set color for piece
    if color == 'black':
        border_color = (255, 255, 255)
        inner_color = (0, 0, 0)
    elif color == 'white':
        border_color = (0, 0, 0)
        inner_color = (255, 255, 255)
    
    pygame.draw.circle(screen, border_color, (posX, posY), 12) # draw piece border
    pygame.draw.circle(screen, inner_color, (posX, posY), 10) # draw piece
    
    # draw king 'status'
    if king == True:
        pygame.draw.circle(screen, border_color, (posX+3, posY-3), 12) # draw piece border
        pygame.draw.circle(screen, inner_color, (posX+3, posY-3), 10) # draw piece

# show message for user on screen
def show_message(message):
    text = font.render(' '+message+' ', True, (255, 255, 255), (120, 195, 46)) # create message
    textRect = text.get_rect() # create a rectangle
    textRect.centerx = screen.get_rect().centerx # center the rectangle
    textRect.centery = screen.get_rect().centery
    screen.blit(text, textRect) # blit the text

# show countdown on screen
def show_countdown(i):
    while i >= 0:
        tim = font_big.render(' '+repr(i)+' ', True, (255, 255, 255), (20, 160, 210)) # create message
        timRect = tim.get_rect() # create a rectangle
        timRect.centerx = screen.get_rect().centerx# center the rectangle
        timRect.centery = screen.get_rect().centery +50
        screen.blit(tim, timRect) # blit the text
        pygame.display.flip() # display scene from buffer
        i-=1
        pygame.time.wait(1000) # pause game for a second

# will display the winner and do a countdown to a new game
def show_winner(winner):
    global board # we are resetting the global board

    if winner == 'draw': show_message("draw, "+str(NN.black)+" - "+str(NN.draw)+" - "+str(NN.white))
    else: show_message(winner+" wins, "+str(NN.black)+" - "+str(NN.draw)+" - "+str(NN.white))
    pygame.display.flip() # display scene from buffer
    show_countdown(pause) # show countdown for number of seconds
    board = init_board() # ... and start a new game

# function displaying position of clicked square
def mouse_click(pos):
    global selected, move_limit # use global variables

    # only go ahead if we can actually play :)
    if (turn != 'black' and white.type != 'cpu') or (turn != 'white' and black.type != 'cpu'):
        column = int(pos[0]/(window_size[0]/board_size))
        row = int(pos[1]/(window_size[1]/board_size))

        if board[row][column] != 0 and board[row][column].color == turn:
            selected = row, column # 'select' a piece
        else:
            moves = avail_moves(board, turn) # get available moves for that player
            for i in range(len(moves)):
                if selected[0] == moves[i][0] and selected[1] == moves[i][1]:
                    if row == moves[i][2] and column == moves[i][3]:
                        make_move([np.concatenate((selected, (row, column)))],0, board) # make the move
                        
                        for m in range(i,len(moves)):
                            if (moves[m][0],moves[m][1])==(moves[i][2],moves[i][3]):
                                move_limit[1] += 1
                                return
                        move_limit[1] += 1 # add to move limit
                        end_turn() # end turn

######################## START OF GAME ########################

pygame.init() # initialize pygame

board = game_init('') # initialize players and board for the game
NN.convert_board(board)
#player_check() # will check for errors in player settings
ply_check() # make changes to player's ply if playing vs human

screen = pygame.display.set_mode(window_size) # set window size
pygame.display.set_caption(title) # set title of the window
clock = pygame.time.Clock() # create clock so that game doesn't refresh that often

background = pygame.image.load(background_image_filename).convert() # load background
font = pygame.font.Font('freesansbold.ttf', 11) # font for the messages
font_big = pygame.font.Font('freesansbold.ttf', 13) # font for the countdown

show=0
while True: # main game loop
    for event in pygame.event.get(): # the event loop
        if event.type == QUIT:
            exit() # quit game
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == left:
            mouse_click(event.pos) # mouse click
        elif event.type == pygame.KEYDOWN:
            show=0
            if event.key == pygame.K_F1: # when pressing 'F1'...
                black.ply_depth=black.ply_depth+1
                board = game_init(" ")
            if black.ply_depth!=1:
                if event.key == pygame.K_F2: # when pressing 'F2'...
                    black.ply_depth=black.ply_depth-1
                    board = game_init(" ")
            if event.key == pygame.K_F3: # when pressing 'F1'...
                white.ply_depth=white.ply_depth+1
                board = game_init(" ")
            if white.ply_depth!=1:
                if event.key == pygame.K_F4: # when pressing 'F2'...
                    white.ply_depth=white.ply_depth-1
                    board = game_init(" ")
            if event.key == pygame.K_F5: # when pressing 'F3'...
                board = game_init('human')
            if event.key == pygame.K_F6: # when pressing 'F3'...
                board = game_init('minimax')
                
    screen.blit(background, (0, 0)) # keep the background at the same spot
    
    # let user know what's happening (whose turn it is)
    # create antialiased font, color, background
    if show<3:
        show_message(str(black.ply_depth)+" - " +str(white.ply_depth))
        show=show+1
    else:
        if white.type=="human" or black.type =="human":
            if (turn != 'black' and white.type == 'human') or (turn != 'white' and black.type == 'human'): show_message('YOUR TURN')
            else: show_message('CPU THINKING...')
    
    # draw pieces on board
    for m in range(8):
        for n in range(8):
            if board[m][n] != 0:
                draw_piece(m+1, n+1, board[m][n].color, board[m][n].king)

    # show intro
    if start == True:
        show_message('Welcome to '+title)
        show_countdown(pause)
        start = False

    # check state of game
    end = end_game(board)
    if end[1] == 0:
        NN.black=NN.black+1
        NN.turn=0
        show_winner("black")
        turn = "white"
        printResult()
    elif end[0] == 0:
        NN.white=NN.white+1
        NN.turn=0
        show_winner("white")
        turn="black"
        printResult()

    # check if we breached the threshold for number of moves    
    elif move_limit[0] == move_limit[1]:
        NN.draw=NN.draw+1
        show_winner("draw")
        if random.uniform(0,1)<0.5:
            turn = "black"
        else: turn = "white"
        printResult()
    else: pygame.display.flip() # display scene from buffer

    # cpu play   
    if turn != 'black' and white.type == 'cpu': cpu_play(white) # white cpu turn
    elif turn != 'white' and black.type == 'cpu': cpu_play(black) # black cpu turn
    
    clock.tick(fps) # saves cpu time
