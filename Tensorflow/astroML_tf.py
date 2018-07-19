
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

from IPython.display import Image
import matplotlib.pyplot as plt
import scipy.io as sio

import numpy as np
import sys
import os
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import math

#from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
#from astroML.utils import completeness_contamination
#
##----------------------------------------------------------------------
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=False)



#Convert labels from label to CNTK output format, basically an array of 0's with a 1 in the position of the desired label so 9 = [0 0 0 0 0 0 0 0 0 1]
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        assi = labels[i]
        if labels[i] == 10:
            assi = 0
        label[i][assi] = 1.0
    return label

def reBalanceData(x,y):
    filter1 = y==1
    ones = x[np.where(y==1)].copy()
    y_ones = y[np.where(y==1)].copy()
    total = len(y)
    total_one = len(ones)
    multiplier = math.ceil(total/total_one)
    for i in range(multiplier):
        x = np.insert(x,1,ones,axis=0)
        y = np.insert(y,1,y_ones,axis=0)

    ran = np.arange(x.shape[0])
    np.random.shuffle(ran)
    x= x[ran]
    y= y[ran]
    return x,y

#############################################################

#############Data Loading & Conversion######################
def predictionMap(xlim,ylim):

    
    mesh = []
    
    for x in np.arange(xlim[0],xlim[1],0.01):
        for y in np.arange(ylim[0],ylim[1],0.01):
            mesh.append([x,y])
            
    return (np.array(mesh))

X = np.loadtxt('AstroML_Data.txt',dtype=float)
y =  np.loadtxt('AstroML_Labels.txt',dtype=float)
print(X)
#X = X[:, [1, 0, 2, 3]]  # rearrange columns for better 1-color results
X = X[:, [1, 0]]  # rearrange columns for better 1-color results
#X = np.insert(X,X.shape[1],np.multiply(X[:,[0]],X[:,[0]]).flatten(),axis=1)
#X = np.insert(X,X.shape[1],np.multiply(X[:,[1]],X[:,[1]]).flatten(),axis=1)
X_train,y_train = reBalanceData(X,y)
filter1=y_train==1
y_train[filter1] = 0.99 
#(X_train, X_test), (y_train, y_test) = split_samples(X, y, [1, 0],
  #                                                   random_state=0)

features = X_train
#labels =np.matrix(y_train).T
labels = y_train


print(labels)
N_tot = len(y)
#Total assignments of 0 (Classification = true)
N_st = np.sum(y == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st
#Number of train labels
#N_train = len(y_train)
#Number of test labels
#N_test = len(y_test)
N_plot = 5000 + N_rr
fig = plt.figure(figsize=(5, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(111)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()
###########################################################
############Netowork Building##############################
#Define input and output dimensions
model = keras.Sequential([
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.relu)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(features, labels, batch_size=200,epochs=15)


xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)
test = predictionMap(xlim,ylim)

predictions = model.predict(test)
xshape = int((xlim[1]-xlim[0])*100)
yshape = int((ylim[1]-ylim[0])*100)
plt.imshow(np.reshape(predictions,(xshape,yshape)))
plt.show()
print(predictions[predictions<0.2].shape,predictions[predictions>0.8].shape)
