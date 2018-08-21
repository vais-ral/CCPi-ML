# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:57:52 2018

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

#TF Neaural Network Builder--------------------------------------

model = keras.Sequential([
        keras.layers.Dense(400, activation=tf.nn.relu),
        keras.layers.Dense(25, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_features , training_labels, epochs=15)

predictions = model.predict(test_features)

count = 0
for i in range(0, len(test_labels)):
    pred = (np.argmax(predictions[i]))
    if test_labels[i][0] == pred:
        count +=1

print("Correct predictions: ", count)
