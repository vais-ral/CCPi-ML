# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:51:18 2018

@author: lhe39759
"""
import sys
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub/')
from SliceOPy import NetSlice, DataSlice
import keras
import numpy as np
from astroML.utils import completeness_contamination
import matplotlib.pyplot as plt
featCols = [[1,0],[1,0,2],[1,0,2,3]]
nets = []

for col in range(0,3):
    
    model = keras.Sequential([
            keras.layers.Dense(8,input_dim =col+2, activation="sigmoid"),
            keras.layers.Dense(4, activation="sigmoid"),
            keras.layers.Dense(2, activation ="sigmoid")
            ])
    
    
    nets.append(NetSlice.NetSlice(model,'keras'))


X_test_unbalanced = np.load('AstroML_X_Test_Shuffle_Split_0_7.npy')
y_test_unbalanced = np.load('AstroML_Y_Test_Shuffle_Split_0_7.npy')
cont = []
comp = []

for col in range(0,1):
    
    data = DataSlice.DataSlice(Features = None,Labels = None)
    data.loadFeatTraining( np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadFeatTest( np.load('AstroML_X_Test_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadLabelTraining( np.load('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.loadLabelTest( np.load('AstroML_Y_Test_Shuffle_Split_0_7_Rebalance_1.npy'))
    data.featureColumn(featCols[col])

    print(data.X_train.shape,data.y_train.reshape(data.y_train.shape[0],1).shape)
    trainFourier = np.fft.fft(np.append(data.X_train,data.y_train.reshape(data.y_train.shape[0],1),axis=1))
    testFourier = np.fft.fft(np.append(data.X_test,data.y_test.reshape(data.y_test.shape[0],1),axis=1))

    data.loadLabelTraining(trainFourier[:,[0,1]])
    data.loadLabelTest(testFourier[:,[0,1]])

    nets[col].loadData(data)
    #nets[col].loadModel('astro_sliceOPy_'+str(col),None)
    print(nets[col].summary())
    routineSettings = {"CompileAll":True, "SaveAll":None}

    trainRoutine = [{"Compile":[keras.optimizers.Adam(lr=0.1),'mean_squared_error',['binary_accuracy', 'categorical_accuracy']],
                "Train":[1000,None,0]},{"Compile":[keras.optimizers.Adam(lr=0.01),'mean_squared_error',['binary_accuracy', 'categorical_accuracy']],
                "Train":[1000,None,0]},{"Compile":[keras.optimizers.Adam(lr=0.001),'mean_squared_error',['binary_accuracy', 'categorical_accuracy']],
                "Train":[1000,None,0]},{"Compile":[keras.optimizers.Adam(lr=0.0001),'mean_squared_error',['binary_accuracy', 'categorical_accuracy']],
                "Train":[1000,None,1]}]
    
    trainRoutine = [{"Compile":[keras.optimizers.Adam(lr=0.1),'mean_squared_error',['binary_accuracy', 'categorical_accuracy']],
                "Train":[100,None,1]}]

    nets[col].trainRoutine(routineSettings,trainRoutine)
    predictions = np.around(nets[col].model.evaluate(data.X_test[:,featCols[col]],data.y_test[:,featCols[col]]))
    print('Test loss:', predictions[0])
    print('Test accuracy:', predictions[1])
    #completeness, contamination = completeness_contamination(predictions,(y_test_unbalanced))
    cont.append(contamination)
    comp.append(completeness)
    #nets[col].saveModel('astro_sliceOPy_'+str(col))
    nets[col].plotLearningCurve(Loss_Val_Label=str(col)+"Cross",Loss_label=str(col))
    if col == 0:
        nets[col].contourPlot()
#%%

for i in range(0,3):
    print(i,"Completeness",comp[i],"Contamination",cont[i])