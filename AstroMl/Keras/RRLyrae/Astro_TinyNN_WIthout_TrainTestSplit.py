# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:23:30 2018

@author: zyv57124
"""

import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Load data ------------------------------------------------------
def loadMATData(file1):
    return sio.loadmat(file1)

#Load Data-------------------------------------------------------

features = np.loadtxt('AstroML_Data.txt', dtype = int)
labels = np.loadtxt('AstroML_Labels.txt', dtype = int)
print (len(features))

filter1 = labels == 1
labels[filter1] = 0.99


training_features = features
training_labels = labels
test_features = features
test_labels = labels

#TF Neaural Network Builder--------------------------------------

model = keras.Sequential([
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.softmax)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_features , training_labels, epochs=15)

predictions = model.predict(test_features)
print(len(predictions))

count = 0
for i in range(len(features)):
    pred = (np.argmax(predictions[i]))
    if test_labels[i] == 1:
        print ('yay')
    if test_labels[i] == pred:
        count +=1

print("Correct predictions: ", count)