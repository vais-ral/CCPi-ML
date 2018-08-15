# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:51:18 2018

@author: lhe39759
"""
import sys
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML/')
from SliceOPy import NetSlice, DataSlice
import keras

featCols = [[1,0],[1,0,2],[1,0,2,3]]
nets = []

for col in range(0,3):
    
    model = keras.Sequential([
            keras.layers.Dense(26,input_dim =col+2, activation="tanh"),
            keras.layers.Dense(1, activation ="sigmoid")
            
            ])
    
    
    nets.append(NetSlice.NetModel(model,'keras'))
    
for col in range(0,3):
    
    data = DataSlice.NetData(Features = None,Labels = None)
    data.loadFeatTraining( np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadFeatTest( np.load('AstroML_X_Test_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadLabelTraining( np.load('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadLabelTest( np.load('AstroML_Y_Test_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.featureColumn(featCols[col])
    nets[col].loadData(data)
    #nets[col].loadModel('astro_sliceOPy_'+str(col)+'.h5',None)
    nets[col].compileModel(keras.optimizers.Adam(lr=0.003),'binary_crossentropy', ['binary_accuracy', 'categorical_accuracy'])
    nets[col].trainModel(Epochs = 1000, Verbose = 2)

#%%
    
for i in range(0,3):
    nets[i].saveModel('astro_sliceOPy_'+str(i)+'.h5')
#%%
    
nets[0].contourPlot()
    