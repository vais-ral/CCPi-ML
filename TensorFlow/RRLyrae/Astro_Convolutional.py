# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:12:03 2018

@author: zyv57124
"""

import matplotlib
matplotlib.use("Agg")
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
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle

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
EPOCHS = 100   #Set epochs
INIT_LR = 0.003   #Set learning rate
IMAGE_DIMS = (20,20,1)  #set image dimensions
num_classes = 1

Data_Astro = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
Labels_Astro = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)
Data_Astro = Data_Astro[:, [1, 0]]

print(Data_Astro.shape)

X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)
print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
N_tot=len(y_train)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr

#Plot original data
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

N_tot=len(y_train)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing


#Weighting
filter1=y_train==0
y_train[filter1] = 0
filter1=y_train==1
y_train[filter1] = 1
X_train,y_train = reBalanceData(Data_Astro,Labels_Astro)

img_x, img_y = 20, 20

input_shapes = (20,20, 1)
# convert the data to the right type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Build model
class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

class SmallerVGGNet:
	def build(width, height, depth, classes):
            model = Sequential()
            model.add(Conv2D(32, (2, 2), padding="same", input_shape=input_shapes))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=1))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.25))
            return model

model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2], classes='2')

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
#
cb=TimingCallback()

history = model.fit(X_train, y_train, batch_size=BS,validation_data=(X_test, y_test),epochs=EPOCHS, verbose=1)
K.get_session().graph
## save the model to disk
#print("[INFO] serializing network...")
#model.save(args["model"])
#
## plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
#N = EPOCHS
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="upper left")
#plt.savefig(args["plot"])

#
#loss_data = (history.history['loss'])
#
#a = np.transpose(model.predict(X_test))
#
#
#xlim = (0.7, 1.35)
#ylim = (-0.15, 0.4)
#
#mesh = predictionMap(xlim,ylim)   #makes mesh array
#xshape = int((xlim[1]-xlim[0])*1000)+1
#yshape = int((ylim[1]-ylim[0])*1000)
#predictions = model.predict(mesh[:,[1,0]])  #classifies points in the mesh 1 or 0
##%%
#fig = plt.figure(figsize=(5, 2.5))
#fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
#ax = fig.add_subplot(1, 1, 1)
#im=ax.scatter(X_test[:, 1], X_test[:, 0], c=a[0], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
#im.set_clim(-0.5, 1)
#ax.contour(np.reshape(mesh[:,0], (xshape, yshape)), np.reshape(mesh[:,1],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)), cmap=plt.cm.binary,lw=2)
#plt.show()

