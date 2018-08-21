# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:35:46 2018

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
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)
    
#############Data Loading & Conversion######################
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


BS = 1000   #Set batch size
EP = 100   #Set epochs
LR = 0.003   #Set learning rate

Data_Astro = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
Labels_Astro = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)
Data_Astro = Data_Astro[:, [1, 0]]

N_tot=len(Labels_Astro)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr

fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()
#%%
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
Data_Astro,Labels_Astro = reBalanceData(Data_Astro,Labels_Astro)

X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)
#Weighting

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

#Build model
class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

# size of image in pixel
img_rows, img_cols = 500, 500
# number of classes (here digits 1 to 10)
nb_classes = 2
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
nb_pool = 20
# convolution kernel size
nb_conv = 20

X = np.vstack([X_train, X_test]).reshape(-1, 1, img_rows, img_cols)
y = np_utils.to_categorical(np.concatenate([y_train, y_test]), nb_classes)

# build model
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# run model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

cb=TimingCallback()
history = model.fit(X_train, y_train, batch_size=BS, epochs = EP)

loss_data = (history.history['loss'])
print(loss_data)

a = np.transpose(model.predict(X_test))


xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)

mesh = predictionMap(xlim,ylim)   #makes mesh array
xshape = int((xlim[1]-xlim[0])*1000)+1
yshape = int((ylim[1]-ylim[0])*1000)
predictions = model.predict(mesh[:,[1,0]])  #classifies points in the mesh 1 or 0
#%%
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(X_test[:, 1], X_test[:, 0], c=a[0], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
ax.contour(np.reshape(mesh[:,0], (xshape, yshape)), np.reshape(mesh[:,1],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)), cmap=plt.cm.binary,lw=2)
plt.show()

