# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:10:47 2018

@author: zyv57124
"""

import scipy.io as sio
import tensorflow as tf
import tensorflow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History
history = History()

#Load data ------------------------------------------------------
def loadMATData(file1):
    return sio.loadmat(file1)

#Load Data-------------------------------------------------------

data = loadMATData('ex3data1.mat')
features = data['X']
labels = data['y']

filter = labels ==10
labels[filter] = 0

#shuffle data---------------------------------------------------

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

#Keras neural network builder------------------------------------

#Reshape data
training_features = training_features.reshape(training_features.shape[0], 1, 20, 20)
test_features = test_features.reshape(test_features.shape[0], 1, 20, 20)
training_features = training_features.astype('float32')
test_features = test_features.astype('float32')

model = Sequential()     #Declare sequential model format

#convolution2D(no. of conv filters, no. of rows in each kernel, no. of columns in each kernel). Input shape=(depth, dims)
model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(1, 20, 20)))   #Declare the input layer
model.add(MaxPooling2D(pool_size=(1,1)))   #Add more layers
model.add(Dropout(0.25))     #Dropout layer regualarises model, prevents overfitting
model.add(Flatten())       #weights must be made 1D before being passed to dense layer
model.add(Dense(128, activation='relu'))   #Dense layer is fully connected. Argument is output size of layer
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(training_features.shape)
print(test_features.shape)
print(training_features.shape)
print(test_features.shape)