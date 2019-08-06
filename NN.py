import tensorflow as tf
from tensorflow import keras
from DataBase import get_direc
print(tf.__version__)

import random
import math
import h5py
import DataBase as db
import numpy as np


turn=0
draw=0
black=0
white=0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(8, 4)),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(500, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def get_data(train_size,test_size,data):
    (train_states,train_plays)=data
    (test_states,test_plays)=(train_states[train_size:train_size+test_size],train_plays[train_size:train_size+test_size])
    (train_states,train_plays)=(train_states[:train_size],train_plays[:train_size])
    return train_states,train_plays,test_states,test_plays
    
def load_model(test_states,test_plays,checkpoint_path,model):
    model.load_weights(checkpoint_path)


def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 8, 4))
    batch_labels = np.zeros((batch_size,128))
    while True:
        for i in range(batch_size):
        # choose random index in features
            index= random.randrange(78800)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        batch_features=tf.one_hot(batch_features, 4)
        batch_labels=tf.one_hot(batch_labels, 1)
        yield batch_features, batch_labels
   
def train_model(numEpochs,train_states,train_plays,test_states,test_plays,cp_callback,model):
    model.fit(train_states, train_plays,
        validation_data = (test_states,test_plays),
        steps_per_epoch=250,
        validation_steps=250,
        epochs=numEpochs,
        shuffle=True,
        callbacks=cp_callback)  # pass callback to training

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    print (lrate)
    return lrate

def see_analyse(test_states,test_plays,model):
    predictions = model.predict(test_states)
#print(test_states[900])
    rge=len(test_plays)
    j=0
    for i in range(rge):
        l=np.argsort(np.argsort(predictions[i]))
        print(127-l[np.argmax(test_plays[i])],convert_move(np.argmax(test_plays[i]),0))
        if 127-l[np.argmax(test_plays[i])]>2:
            j=j+1
    print("acc:",(rge-j)/rge)

def predict(board, avail,model,rangeM=3):
    moves=np.empty((0,4),dtype=int)
    m=0
    board=convert_board(board)
    prediction=model.predict(np.reshape([board], (1,8,4)))
    pred_ranked=np.argsort(np.argsort(prediction))
    piece_avail = avail[:,:2]
    direc_avail=np.empty((len(avail),1))
    
    for i in range(len(avail)):
        aux=np.r_[convert_move(db.to_database_index(avail[i][0],avail[i][1],0,8),2,8),
        convert_move(db.to_database_index(avail[i][2],avail[i][3],0,8),2,8)]
        direc_avail[i]=db.get_direc((aux[0],aux[2]), (aux[1],aux[3]))
    move_avail=np.c_[piece_avail,direc_avail]
    
    for i in range(127,-1,-1):
        piece_predic,direc_predic=convert_move(np.where(pred_ranked==i)[1][0], 1)
        move_predic=np.r_[piece_predic,direc_predic]
        if any([np.array_equal(move_predic, a) for a in move_avail]) and m<rangeM:
            moves= np.r_['0,2',moves,avail[np.where([np.array_equal(move_predic, a) for a in move_avail])[0][0]]]
            m=m+1
    return moves

def convert_move(move,type,nb_col=4):
    direc=move%4
    bruto=int((move-direc)/4)
    m=int(bruto/nb_col)
    n=bruto%nb_col
    
    if type==0: #converte para estilo board real
        if direc==0:
            direc='++'
        elif direc==1:
            direc='+-'
        elif direc==2:
            direc='-+'
        else:
            direc='--'
        return bruto+1,direc
    
    elif type==1: #converte para estilo board Checkers
        if m %2== 1:
            n=n*2
        else:
            n=n*2+1
        return [m,n],direc
    elif type==2: #converte para estilo board Database
        try:
            if m %2== 1:
                n=n/2
            else:
                n=(n-1)/2
        except:
            n=0
        return [m,int(n)]

def convert_board(board):
    new_board=np.zeros((8,4))
    for m in range(8):
        i=0
        for n in range(0 if m%2!=0 else 1,8,2):
            try:
                if board[m][n].color=="black":
                    if board[m][n].king:
                        new_board[m][i]=3
                    else:
                        new_board[m][i]=1
                else:
                    if board[m][n].king:
                        new_board[m][i]=-3
                    else:
                        new_board[m][i]=-1
            except:
                new_board[m][i]=0
            i=i+1

    return new_board

def init_callback(redesNB):
    early_stop=keras.callbacks.EarlyStopping(monitor='val_lost',
                              min_delta=1,
                              patience=50,
                              verbose=1, mode='auto')

    learning_rate_decay=keras.callbacks.LearningRateScheduler(step_decay)
    
    
    
    
    
    if(redesNB==0):
        checkpoint = keras.callbacks.ModelCheckpoint("./training_1/cp.ckpt",mode='min',monitor='val_loss',save_weights_only=True,verbose=1)
        mcp_save = keras.callbacks.ModelCheckpoint("./training_1/cp.mdl_wts.hdf5", save_best_only=True, monitor='val_loss', mode='min')
        mcp_save_acc = keras.callbacks.ModelCheckpoint("./training_1/acc_cp.mdl_wts.hdf5", save_best_only=True, monitor='val_acc', mode='auto')
    else:
        checkpoint = keras.callbacks.ModelCheckpoint("./training_2/cp.ckpt",mode='min',monitor='val_loss',save_weights_only=True,verbose=1)
        mcp_save = keras.callbacks.ModelCheckpoint("./training_2/cp.mdl_wts.hdf5", save_best_only=True, monitor='val_loss', mode='min')
        mcp_save_acc = keras.callbacks.ModelCheckpoint("./training_2/acc_cp.mdl_wts.hdf5", save_best_only=True, monitor='val_acc', mode='auto')
    return [mcp_save,mcp_save_acc,checkpoint,learning_rate_decay]

