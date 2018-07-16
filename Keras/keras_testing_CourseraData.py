# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#Compile Model-------------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   #Defines loss function and optimizer

#Fit model on training data------------------------------------------------

history = model.fit(training_features, training_labels, batch_size=32, nb_epoch=100, verbose=1, callbacks=[history])
print (history.history.keys())
print (history.history['loss'])


#Handmade plot of loss against epoch---------------------------------------

Losses = np.array(history.history['loss'])
Iterations = np.arange(0,100,1)

plt.plot(Iterations, Losses)
plt.show()
#---------------------------------------------------------------------------

score = model.evaluate(test_features, test_labels, verbose=2)  #Testing

#Plot of loss against epoch using Keras features ---------------------------
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()