# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:37:13 2018

@author: zyv57124
"""

import scipy.io as sio
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History
from keras import optimizers


#Timing-----------------------------------------------------------------
from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)


#Load data ---------------------------------------------------------------
def loadMATData(file1):
    return sio.loadmat(file1)

#Load Data----------------------------------------------------------------

data = loadMATData('ex3data1.mat')
features = data['X']
labels = data['y']

filter = labels ==10
labels[filter] = 0

#shuffle data-------------------------------------------------------------

ran = np.arange(features.shape[0])
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]

training_features = features[:3500]
training_labels = labels[:3500]
test_features = features[3501:]
test_labels = labels[3501:]

#convert 1D class arrays to 10D class matrices
training_labels = np_utils.to_categorical(training_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

#Keras neural network builder---------------------------------------------

#Reshape data
training_features = training_features.astype('float32')
test_features = test_features.astype('float32')

#-------------------------------------------------------------------------
for i in np.arange(0,500, 10):
    model = Sequential()     #Declare sequential model format
    model.add(Dense(400, activation='relu')) #Declare the input layer
    model.add(Dense(10, activation='softmax'))
    
    #Compile Model-------------------------------------------------------------
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])   #Defines loss function and optimizer

    #Fit model on training data------------------------------------------------
    cb=TimingCallback()
    history = model.fit(training_features, training_labels, batch_size=i+1, nb_epoch=10, verbose=2, callbacks=[cb])


#print (history.history.keys())
    print (cb)

    #Store eoch number and loss values in .txt file
    loss_data = (history.history['loss'])
    f = open("loss_data_batchnum_"+str(i+1)+".txt","w")
    for xx in range(1,len(loss_data)+1):
        if xx==1:
            delta_loss = 'Nan'
        else:
            delta_loss = (loss_data[xx-2] - loss_data[xx-1])
        f.write(str(xx) + "," + str(loss_data[xx-1]) + "," + str(i+1) + "," + str(cb.logs[xx-1]) + "," + str(delta_loss) + "\n" )
    f.close()


#print(range(0,len(loss_data)))
score = model.evaluate(test_features, test_labels, verbose=2)  #Testing
