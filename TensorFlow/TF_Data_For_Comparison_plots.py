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
from tensorflow.python.training import gradient_descent


from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)
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

for i in np.arange(0,500, 10):
    #TF Neaural Network Builder--------------------------------------
    
    model = keras.Sequential([
            keras.layers.Dense(400, activation=tf.nn.relu),
            keras.layers.Dense(25, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
  
    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    predictions = model.predict(test_features)
    
    cb=TimingCallback()
    history = model.fit(training_features, training_labels, batch_size=i+1, epochs=100, verbose=2, callbacks=[cb])
   
    #Store eoch number and loss values in .txt file
    loss_data = (history.history['loss'])
    f = open("TF_loss_data_batchnum_"+str(i+1)+".txt","w")
    for xx in range(1,len(loss_data)+1):
        if xx==1:
            delta_loss = 'Nan'
        else:
            delta_loss = (loss_data[xx-2] - loss_data[xx-1])
            #Epoch                   #Loss                  #Batch size          #Time                  #Change in loss
        f.write(str(xx) + "," + str(loss_data[xx-1]) + "," + str(i+1) + "," + str(cb.logs[xx-1]) + "," + str(delta_loss) + "\n" )
    f.close()

