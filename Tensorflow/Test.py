# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadMATData(file1):
    return sio.loadmat(file1)

features = np.loadtxt('AstroML_Data.txt',dtype=int)
labels =  np.loadtxt('AstroML_Labels.txt',dtype=int)
features = features[:, [1, 0,3,4]]



#feat_train = features[:3500]
#labels_train = labels[:3500]
#feat_test = features[3501:]
#labels_test = labels[3501:]

model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(features, labels, epochs=15)

#predictions = model.predict(feat)




