# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:17:25 2018

@author: zyv57124
"""

import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import sklearn
from tensorflow import keras
from sklearn.model_selection import train_test_split
import math
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from astroML.utils import completeness_contamination

##############Timing fucntion###############################
from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)
    
#############Data Loading & Weighting########################
def predictionMap(xlim,ylim):    
    mesh = []
    
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y])
            
    return (np.array(mesh))

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

Data_Astro = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
Labels_Astro = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)
Data_Astro = Data_Astro[:, [1, 0]]

N_tot=len(Labels_Astro)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr

#%%
############################Plot Data#########################
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.ylabel('g-r')
plt.xlabel('u-g')
ax.set_title('Classified Stars')
plt.show()

#%%

#######################Prepare Data for model####################
Data_Astro,Labels_Astro = reBalanceData(Data_Astro,Labels_Astro)

BS = 1000   #Set batch size
EP = 100   #Set epochs
LR = 0.01   #Set learning rate

# Set variables for ReBalance Function
filter1=y_train==0
y_train[filter1] = 0
filter1=y_train==1
y_train[filter1] = 1
#Split data into training and testing samples
Data_Astro,Labels_Astro = reBalanceData(Data_Astro,Labels_Astro)
X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
#class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

#%%
#########################Build model##############################


model = Sequential()
model.add(Dense(8, input_dim=2, kernel_initializer='normal', activation='tanh'))  #tanh -1<f<1
model.add(Dense(20, activation='tanh'))  
model.add(Dense(1, activation='sigmoid'))    #Sigmoid 0<f<1
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb=TimingCallback()
history = model.fit(X_train, y_train, batch_size=BS, epochs = EP)

loss_data = (history.history['loss'])
print(loss_data)
print(cb)
a = np.transpose(model.predict(X_test))

#####################################################################
#Make meshgrid same shape as plot and classify every point in grid as 1 or 0

xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)
mesh = predictionMap(xlim,ylim)   #makes mesh array
xshape = int((xlim[1]-xlim[0])*1000)+1
yshape = int((ylim[1]-ylim[0])*1000)
predictions = model.predict(mesh[:,[1,0]])  #classifies points in the mesh 1 or 0
#%%

#############Plot decision boundary over weighted datd################
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(X_test[:, 1], X_test[:, 0], c=a[0], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
ax.contour(np.reshape(mesh[:,0], (xshape, yshape)), np.reshape(mesh[:,1],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)), cmap=plt.cm.binary,lw=2)
plt.ylabel('g-r')
plt.xlabel('u-g')
ax.set_title('Decison Boundary Over Pre-Classified Stars')
plt.show()

#%%

##########################Evaluate Perforamce#######################

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
ax.set_title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
ax.set_title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#e = np.array(history.history['acc'])
#print (e.shape)
#for i in np.arange(1,100,1):
#        plt.plot(i,(math.log(e[i])))      
#plt.show()



