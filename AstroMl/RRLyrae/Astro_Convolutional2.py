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
from keras.layers.convolutional import Conv1D
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
#%%
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

#Load data ------------------------------------------------------

features = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
labels = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)
features = features[:, [1, 0]]
#Load Data-------------------------------------------------------

#shuffle data---------------------------------------------------

ran = np.arange(len(features))
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]

training_features = features[:74512]
training_labels = labels[:74512]
test_features = features[74513:]
test_labels = labels[74513:]

N_tot=len(Labels_Astro)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr

#training_features = training_features.astype('float32') 
#test_features = test_features.astype('float32')
#training_labels = training_labels.astype('float32') 
#test_labels = test_labels.astype('float32')
#training_features /= np.max(training_features) # Normalise data to [0, 1] range
#test_features /= np.max(test_features) # Normalise data to [0, 1] range

#training_labels = np_utils.to_categorical(training_labels, num_classes) # One-hot encode the labels
#test_labels = np_utils.to_categorical(test_labels, num_classes) # One-hot encode the labels

#Plot data
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

#%%
training_labels = labels[:74512]
test_features = features[74513:]
test_labels = labels[74513:]

num_epochs = 200
kernel_size = 3
pool_size = 2

model = Sequential()
model.add(Conv1D(6, 2, padding="valid"))
model.add(Dense(2))
model.add(Activation('sigmoid'))
    
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_features , training_labels, epochs=15)

predictions = model.predict(test_features)

count = 0
for i in range(0, len(test_labels)):
    pred = (np.argmax(predictions[i]))
    if test_labels[i] == pred:
        count +=1

print("Correct predictions: ", count, " / 18629" )
print (float(int(count) / int(18629)))
