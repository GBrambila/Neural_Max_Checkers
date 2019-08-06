import numpy as np
import re
#61044


def getBoardLoad(string):
    data_input=load("data_input"+string,3)
    data_output=load("data_output"+string,2)
    return data_input,data_output

def load(file,dim):
    with open(file,"r") as f:
        line=f.readline()
        line=re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]|\n',line)
        while('' in line): 
            line.remove('') 
        line = list(map(int, line))
        if dim==3:
            shape=(line[0],line[1],line[2])
        elif dim==2:
            shape=(line[0],line[1])
        return load_data(file,shape)

def get_board(nb_moves,nb_matchs):
    get_matchs()
    data_input = np.zeros((nb_moves,20000,8,4))
    data_output = np.zeros((nb_moves,20000,128))
    
    
    for match in range(nb_matchs):
        try:
            board = init_board()
            moves=get_moves(match)[:nb_moves]
        
            make_move(board, treat_move(moves[0]),0)
        
            for move in range(1,len(moves)):
                board=np.negative(np.flip(board))
                if(move%2==0 and nb_damas(board)>0):
                    data_input[move][match]=board
                    data_output[move][match]=make_move(board, treat_move(moves[move]),move)
                else:
                    make_move(board, treat_move(moves[move]),move)
        except:
            continue
    
    
    for i in range(nb_moves):
        indexes=[j for j in np.arange(16000) if sum_board(data_input[i][j])!=0]
        aux_in= data_input[i][indexes]
        aux_out=data_output[i][indexes]
        aux_in=np.reshape(aux_in,(len(aux_in),8,4))
        aux_out=np.reshape(aux_out,(len(aux_in),128))
        save_data(aux_in,"data_input_"+str(i)+".txt")
        save_data(aux_out,"data_output_"+str(i)+".txt")

def init_board():
    board = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1,-1,-1,-1],
        [-1,-1,-1,-1],
        [-1,-1,-1,-1]
        ]
    return board
    
def make_move(board, move,index):
    len_moves=2 if move[-1]==0 else move[-1]+1
    m=np.zeros(len_moves,dtype=int) #Acho que nunca tera 8 pecas comidas em um movimento
    n=np.zeros(len_moves,dtype=int)
    if index%2!=0:
        for i in range (len_moves):
            move[i]=33-move[i]
    #convert 1d index to 2d
    for i in range(len_moves):
        piece=0
        m[i]
        for m[i] in range(8):
            for n[i] in range(4):
                piece=piece+1
                if piece==move[i]:
                    break
            if piece==move[i]:
                break
    
    if move[-1]==0:
        board[m[1]][n[1]]=board[m[0]][n[0]]
        board[m[0]][n[0]]=0
    else:
        make_jumps(board,m,n)
        board[m[-1]][n[-1]]=board[m[0]][n[0]]
        board[m[0]][n[0]]=0
    
    check_damas(board,m[-1],n[-1])
    
    direc=get_direc(m, n)
    
    output=np.zeros(128)
    output[to_database_index(m[0],n[0],direc,4)]=1
    ''' 
    + + = 0
    + - = 1
    - + = 2
    - - = 3  
    '''
    
    return output

def make_jumps(board,m,n):
    for i in range(len(m)-1):
        coordM=m[i+1]-m[i]
        coordN=n[i+1]-n[i]
        if coordM<0:
            coordM=coordM+1
        else:
            coordM=coordM-1
        board[m[i]+coordM][n[i]+coordN]=0
    
def check_damas(board,m,n):
    if m==0 and board[m][n]==-1:
        board[m][n]=-3
    elif m==7 and board[m][n]==1:
        board[m][n]=3
    
def nb_damas(board):
    damas=0
    for m in range (8):
        for n in range(4):
            if np.abs(board[m][n])==3:
                damas=damas+1
    return damas

def sum_board(board):
    pecas=0
    for m in range (8):
        for n in range(4):
            pecas=pecas+np.abs(board[m][n])
    return pecas
        

def get_direc(m,n):
    if m[-1]-m[0]>0:
        if m[0]%2==0:
            if n[-1]-n[0]>0:
                return 0
            else:
                return 1
        else:
            if n[-1]-n[0]==0:
                return 0
            else:
                return 1
    else:
        if m[0]%2==0:
            if n[-1]-n[0]>0:
                return 2
            else:
                return 3
        else:
            if n[-1]-n[0]==0:
                return 2
            else:
                return 3

def to_database_index(m,n,direc,col_nb):
    return (m*col_nb+n)*4+direc
def get_matchs():
    global matchs
    with open('Matchs.pdn.txt', 'r',encoding='latin-1') as file:
        data = file.read()
        
    matchs=re.split('\[Event|\n1. |\{|\}',data)
    del matchs[0]
    
    i=0
    while i<len(matchs):
        if matchs[i][0:5]!='11-15':
            del matchs[i]
            continue
        matchs[i]=matchs[i].replace('\n',' ')
        i=i+1
    file.close()

def get_moves(match):
    try:
        moves=matchs[match].split()
    except:
        return
    i=0
    while i<len(moves):
        if moves[i].find('.')!=-1: #deletar onde tiver ponto
            del moves[i]
            continue
        i=i+1
        
    return moves

def treat_move(move):
    x=move.count("x")
    move=re.split('-|x',move)
    move=np.r_[move,x]
    return move.astype(int)

def save_data(data,name):
    with open(name,'w') as outfile:
            outfile.write('#{0}\n'.format(data.shape))
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New slice\n')

def load_data(name,shape):
    return np.loadtxt(name).reshape(shape)