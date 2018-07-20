# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:17:25 2018

@author: zyv57124
"""

import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import sklearn
from tensorflow import keras
from sklearn.model_selection import train_test_split

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)

BS = 10   #Set batch size
EP = 4   #Set epochs
LR = 0.001   #Set learning rate

Data_Astro = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
Labels_Astro = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)

Data_Astro = Data_Astro[:, [1, 0]]

N_tot=len(Labels_Astro)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr

fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)



model = Sequential()
model.add(Dense(8, input_dim=2
                , init='uniform', activation='sigmoid'))
model.add(Dense(4, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb=TimingCallback()
history = model.fit(X_train, y_train, batch_size=BS, epochs = EP)

loss_data = (history.history['loss'])
print(loss_data)

a = np.transpose(model.predict(X_test))

print (a)


fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(X_test[:, 1], X_test[:, 0], c=a[0], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()