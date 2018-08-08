from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import numpy as np
import tensorflow as tf
import keras
import math
from keras.metrics import categorical_accuracy
from matplotlib.animation import FuncAnimation
from astroML.utils import completeness_contamination
from astroML.utils import split_samples

#Convert labels from label to CNTK output format, basically an array of 0's with a 1 in the position of the desired label so 9 = [0 0 0 0 0 0 0 0 0 1]
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        if labels[i] == 0:
            label[i,:] = np.array([0,1])
        else:
            label[i,:] = np.array([1,0])
    return label

def reBalanceData(x,y,Multip):
    ones = x[np.where(y==1)].copy()
    y_ones = y[np.where(y==1)].copy()
    total = len(y)
    total_one = len(ones)
    multiplier = int(math.ceil((total/total_one)*Multip))
    for i in range(multiplier):
        x = np.insert(x,1,ones,axis=0)
        y = np.insert(y,1,y_ones,axis=0)

    ran = np.arange(x.shape[0])
    np.random.shuffle(ran)
    x= x[ran]
    y= y[ran]
    return x,y


def predictionMap(xlim,ylim):
    mesh = []
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y,0,0])     
    return (np.array(mesh))

def splitdata(X,y,ratio):
    length = X.shape[0]
    return X[:int(length*ratio)],X[:int(length*(1-ratio))],y[:int(length*ratio)],y[:int(length*(1-ratio))]


def generateData():
    X = np.loadtxt('AstroML_Data.txt')
    y =  np.loadtxt('AstroML_Labels.txt')
    
    X,y = reBalanceData(X,y,1)

    ran = np.arange(X.shape[0])
    np.random.shuffle(ran)
    X= X[ran]
    y= y[ran] 
    
    X_train, X_test, y_train, y_test = splitdata(X, y,0.7)

    np.save('AstroML_X_Train_rebalance_1_split_0_7.npy',X_train)    
    np.save('AstroML_X_Test_rebalance_1_split_0_7.npy',X_test)    
    np.save('AstroML_Y_Train_rebalance_1_split_0_7.npy',y_train)    
    np.save('AstroML_Y_Test_rebalance_1_split_0_7.npy',y_test)    
    
    
#%%
############################################
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
keras.backend.tensorflow_backend._get_available_gpus()
############# Settings #####################

LR =0.0001
Epochs = 3000
BatchSize = np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy').shape[0]
Multip = 1

#############################################################

#############Data Loading & Conversion######################

widthD = []
depthD = []
trainLoss = []
testLoss = []    
testComp = []
testCont = []
comp = []
cont = []
color = []
collD = []
for depth in range(0,11,1):
    for width in range(0,25,1):
        for coll in range(4,5):
            

#for coll in range(3,4):
            print('Width:',width," Depth:",depth,"Col:",coll)

            
            X_train = np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy')
            X_test =  np.load('AstroML_X_Test_Shuffle_Split_0_7_Rebalance_1.npy')
            y_train = np.load('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy')
            y_test =  np.load('AstroML_Y_Test_Shuffle_Split_0_7_Rebalance_1.npy')
            X_test_unbalanced = np.load('AstroML_X_Test_Shuffle_Split_0_7.npy')
            y_test_unbalanced = np.load('AstroML_Y_Test_Shuffle_Split_0_7.npy')
       
            N_tot = y_train.shape[0]
            N_st = np.sum(y_train == 0)
            N_rr = N_tot - N_st
            N_plot = 5000 + N_rr

            if coll==2:

                X_train = X_train[:, [1,0]]  # rearrange columns for better 2-color results
                X_test = X_test[:, [1,0]]  # rearrange columns for better 4-color results
                X_test_unbalanced = X_test_unbalanced[:, [1,0]]  # rearrange columns for better 2-color results
            elif coll==3:
                X_train = X_train[:, [1,0,2]]  # rearrange columns for better 3-color results
                X_test = X_test[:, [1,0,2]]  # rearrange columns for better 4-color results
                X_test_unbalanced = X_test_unbalanced[:, [1,0,2]]  # rearrange columns for better 2-color results
        
            elif coll==4:
                X_train = X_train[:, [1,0,2,3]]  # rearrange columns for better 4-color results
                X_test = X_test[:, [1,0,2,3]]  # rearrange columns for better 4-color results
                X_test_unbalanced = X_test_unbalanced[:, [1,0,2,3]]  # rearrange columns for better 2-color results

  
        	#%%
        
        	###########################################################
        	############Netowork Building##############################
        
        
            layers = []
            layers.append(keras.layers.Dense(width,input_dim=coll,kernel_initializer='normal', activation='tanh'))
        	
            for layer in range(1,(depth)):
                layers.append(keras.layers.Dense(width,kernel_initializer='normal', activation='tanh'))
                
            layers.append(keras.layers.Dense(1,kernel_initializer='normal', activation='sigmoid'))
            
            model = keras.Sequential(layers)
        
            model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
            history = model.fit(X_train, y_train, batch_size=BatchSize,epochs=100, verbose=0)
            model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
            history = model.fit(X_train, y_train, batch_size=BatchSize,epochs=2000, verbose=0)
            model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
            history = model.fit(X_train, y_train, batch_size=BatchSize,epochs=Epochs, verbose=0)
            predictions = np.around(model.predict(X_test_unbalanced).reshape(model.predict(X_test_unbalanced).shape[0],))
        
            completeness, contamination = completeness_contamination(predictions,(y_test_unbalanced))
        
            #scores = model.evaluate(X_test,y_test)
            
            lossTest = scores[0]
            collD.append(coll)
            widthD.append(width)
            depthD.append(depth)
            testLoss.append(lossTest)
            trainLoss.append(history.history['loss'][-1])
            comp.append(completeness)
            cont.append(contamination)
            print("completeness",completeness)
            print("contamination", contamination)
            loss_data = history.history['loss']
            epoch_data = np.arange(0,len((loss_data)))
        
            np.save('WidthDepthData\loss_'+str(coll)+'_'+str(width)+'_'+str(depth)+'.npy',np.array([epoch_data,np.array(loss_data)]))
        
    
#            if True :
#        		
#    
#        
#                xlim = (0.7, 1.35)
#                ylim = (-0.15, 0.4)
#        		    
#                test = predictionMap(xlim,ylim)
#        
#                xshape = int((xlim[1]-xlim[0])*1000)+1
#                yshape = int((ylim[1]-ylim[0])*1000)
#        		
#                test = test[:,[1,0]]
#    
#                predictions =(model.predict(test))
    
        		#%%

print(len(widthD))
f = open('WidthDepthData\\'+str(coll)+'_data.txt',"w")
for xx in range(0,len(widthD)):
    f.write(str(collD[xx]) + "," +str(widthD[xx]) + "," + str(depthD[xx]) + "," + str(testLoss[xx]) + "," + str(trainLoss[xx])  + "," + str(comp[xx]) + "," + str(cont[xx]) + "\n" )
f.close()

#%%
model.save('model.h5')


