import DataBase as db
import NN 
import numpy as np
import tensorflow as tf

load=True
train= True
mostrar_resultados=True
divisao=20
numEpochs=5

model=NN.create_model()
model2=NN.create_model()
loss=np.zeros(50)
acc=np.zeros(50)

def resultados():
    if mostrar_resultados:
        for i in range(min,max):
            if redesNB==0:
                (loss[i],acc[i])=model.evaluate(test_board[i*1689:(i+1)*1689], test_move[i*1689:(i+1)*1689],steps=1)
            else:
                (loss[i],acc[i])=model2.evaluate(test_board[(i-min)*1689:(i-min+1)*1689], test_move[(i-min)*1689:(i-min+1)*1689])
        
            
#db.get_board(40,5629)

model1_path="./training_1/cp.mdl_wts.hdf5"
model2_path="./training_2/cp.mdl_wts.hdf5"
if load:
    try:
        model.load_weights(model1_path)
    except:
        a=0
    try:
        model2.load_weights(model2_path)
    except:
        a=0

for redesNB in range (0,1):
    train_board=np.empty((0,8,4))
    train_move=np.empty((0,128))
    test_board=np.empty((0,8,4))
    test_move=np.empty((0,128))
    if redesNB==0:
        checkpoint_path=model1_path
        min=0
        max=divisao
    else:
        min=divisao
        max=40
        checkpoint_path=model2_path

    for i in range(min,max):
        data=db.getBoardLoad("_"+str(i*2)+".txt")
        train_board=np.r_['0,3',train_board,data[0][:3940]]
        train_move=np.r_['0,2',train_move,data[1][:3940]]
        test_board=np.r_['0,3',test_board,data[0][3940:5629]]
        test_move=np.r_['0,2',test_move,data[1][3940:5629]]
    
    train_board=tf.one_hot(train_board, 4)
    test_board=tf.one_hot(test_board,4)
    '''print(np.shape(train_board))
    with tf.Session() as sess:
        print (sess.run(train_board))'''
    
    if max-min!=0:
        if train:
            if redesNB==0:
                NN.train_model(numEpochs,train_board,train_move,test_board,test_move,NN.init_callback(redesNB),model)
            else:
                NN.train_model(numEpochs,train_board,train_move,test_board,test_move,NN.init_callback(redesNB),model2)
        resultados()

for i in range(50):
            print(i,'. loss:',loss[i],'- acc:',acc[i])
    #if load:
    #if train:
        #'''