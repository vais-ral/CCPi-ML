# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:53:54 2018

@author: lhe39759
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as fi
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    guass = fi.gaussian_filter(inp, nsig)/kernlen
    max1 = np.amax(guass)
    gauss = guass/max1
    return gauss

def generateGaussianHill(xmin,xmax,ymin,ymax,spacer,sig):
    
    gauss = gkern2(spacer,sig)
    x =np.arange(xmin,xmax, (np.abs(xmin)+np.abs(xmax))/spacer)
    y = np.arange(ymin, ymax, (np.abs(ymin)+np.abs(ymax))/spacer)
    X, Y = np.meshgrid(x, y)


    features = []
    for x1 in x:
        for y1 in y:
            item = []
            item.append(x1)
            item.append(y1)   
            features.append(np.array(item))

    features = np.array(features)
    labels = gauss.flatten()
    return features, labels


def generateGaussianHillValley(xmin,xmax,ymin,ymax,spacer,sig):
    
    gauss = np.append(gkern2(spacer,9),-1*gkern2(spacer,9),axis=0)
    x =np.arange(xmin,xmax, (np.abs(xmin)+np.abs(xmax))/spacer)
    y = np.arange(ymin, ymax, (np.abs(ymin)+np.abs(ymax))/(2*spacer))
    X, Y = np.meshgrid(x, y)


    features = []
    for x1 in x:
        for y1 in y:
            item = []
            item.append(x1)
            item.append(y1)   
            features.append(np.array(item))

    features = np.array(features)
    labels = gauss.flatten()
    return features, labels


def plotGaussian(labels,xmin,xmax,ymin,ymax,spacerx,spacery,label):
    x =np.arange(xmin,xmax, (np.abs(xmin)+np.abs(xmax))/spacerx)
    y = np.arange(ymin, ymax, (np.abs(ymin)+np.abs(ymax))/spacery)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gaus = labels.reshape(X.shape)
    ax.plot_surface(X, Y, gaus)
    fig.suptitle(label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def lossPlot(loss,label):
    
    epoch = np.arange(0, len(loss))
    plt.plot(epoch,loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


features, labels = generateGaussianHillValley(-5.0,5.0,-5.0,5.0,100,9)
#plotGaussian(labels,-5.0,5.0,-5.0,5.0,100,200,"Hill Valley")

layer1Neurons = [1,2,3,4,5,7,9,12,15,20,30,40,50]
layer2Neurons = [0,1,2,3,4,5,7,9,12,15]


history = []
surface = []
netDetails = []
params = []

for layer2 in layer2Neurons:
    for layer1 in layer1Neurons:
        
        historyitem =  []
        netDetailsItem = []
        layers=[]
        
        if layer2 == 0:
            
            layers.append(keras.layers.Dense(layer1, input_dim = 2, activation="tanh"))
            layers.append(keras.layers.Dense(1, activation="linear"))
        else:
            
            layers.append(keras.layers.Dense(layer1, input_dim = 2, activation="tanh"))
            layers.append(keras.layers.Dense(layer2, activation="tanh"))
            layers.append(keras.layers.Dense(1, activation="linear"))
            
        model = keras.Sequential(layers)
        print(model.summary())
        model.compile(optimizer=keras.optimizers.Adam(lr=0.1), loss='mean_squared_error', metrics=['binary_accuracy', 'categorical_accuracy'])
        history1 = model.fit(features, labels, batch_size=features.shape[0],epochs=100, verbose=0)
        
        model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='mean_squared_error', metrics=['binary_accuracy', 'categorical_accuracy'])
        history2 = model.fit(features, labels, batch_size=features.shape[0],epochs=2000, verbose=0)
        
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mean_squared_error', metrics=['binary_accuracy', 'categorical_accuracy'])
        history3 = model.fit(features, labels, batch_size=features.shape[0],epochs=5000, verbose=0)
        
        historyitem = np.append(np.array(history1.history['loss']),np.array(history2.history['loss']))
        historyitem = np.append(np.array(historyitem),np.array(history3.history['loss']))

        history.append(historyitem)
        z = model.predict(features)
        surface.append(z)
        
        netDetailsItem.append(layer1)
        netDetailsItem.append(layer2)
        params.append(model.count_params())

np.save('history.npy',np.array(history))
np.save('surface.npy',np.array(surface))
np.save('netDetails.npy',np.array(netDetails))
np.save('params.npy',np.array(params))