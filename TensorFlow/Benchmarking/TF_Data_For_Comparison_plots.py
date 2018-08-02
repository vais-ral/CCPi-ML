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
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        assi = labels[i]
        if labels[i] == 10:
            assi = 0
        label[i][assi] = 1.0
        label[i] = np.transpose(label[i])
    return label

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
training_labels = convertLabels(training_labels,training_labels.shape[0],10)
counts = []
for i in np.arange(0,50, 4):
    #TF Neaural Network Builder--------------------------------------
    if i ==0:
        i=1
    model = keras.Sequential([
            keras.layers.Dense(i, input_dim=400, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=None),
            keras.layers.Softmax(axis=0)

    ])


#    loss_fn = keras.losses.categorical_crossentropy(y_true, y_pred)

    model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    
    predictions = model.predict(test_features)
    
    cb=TimingCallback()
    history = model.fit(training_features, training_labels, batch_size=40, epochs=200, verbose=2, callbacks=[cb])
   
    #Store eoch number and loss values in .txt file
    loss_data = (history.history['loss'])
    f = open("Benchmarking\VWidth\TF_loss_data_batchnum_"+str(i)+".txt","w")
    for xx in range(1,len(loss_data)+1):
        if xx==1:
            delta_loss = 'Nan'
        else:
            delta_loss = (loss_data[xx-2] - loss_data[xx-1])
            #Epoch                   #Loss                  #Batch size          #Time                  #Change in loss
        f.write(str(xx) + "," + str(i) + "," + str(loss_data[xx-1]) + "," + str(delta_loss) + "," +str(cb.logs[xx-1]) + "\n" )
    f.close()

    predictions = model.predict(test_features)
#%%
    count = 0
    for i in range(0, len(test_labels)):
        pred = (np.argmax(predictions[i]))
        if test_labels[i][0] == pred:
            count +=1
            
    counts.append(count)
    print(count)
f = open("Benchmarking\VWidth\TF_counter.txt","w")
itter=0
for it in range(0,50,4):
    f.write(str(it)+","+str(counts[itter]))
    itter +=1
    
f.close()