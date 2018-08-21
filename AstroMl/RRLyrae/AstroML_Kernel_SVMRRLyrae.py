# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:56:26 2018

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
from sklearn.cluster import KMeans
import math
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from astroML.utils import split_samples
from astroML.utils import completeness_contamination
from sklearn.utils import class_weight
from astroML.decorators import pickle_results
from sklearn.svm import SVC
from sklearn import metrics


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


BS = 1000   #Set batch size
EP = 100   #Set epochs
LR = 0.001   #Set learning rate

Data_Astro = np.loadtxt('Data\AstroML_Data.txt',dtype=float)
Labels_Astro = np.loadtxt('Data\AstroML_Labels.txt',dtype=float)
Data_Astro = Data_Astro[:, [1, 0]]

N_tot=len(Labels_Astro)
N_st = np.sum(Labels_Astro == 0)
N_rr = N_tot - N_st
N_plot = 5000 +N_rr
Ncolors = np.arange(1, Data_Astro.shape[1] + 1)

fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0, left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(1, 1, 1)
im=ax.scatter(Data_Astro[-N_plot:, 1], Data_Astro[-N_plot:, 0], c=Labels_Astro[-N_plot:], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_train),
#                                                 y_train)
#Weighting
#X_train,y_train = reBalanceData(Data_Astro,Labels_Astro)
X_train, y_train = Data_Astro, Labels_Astro
filter1=y_train==0
y_train[filter1] = 0
filter1=y_train==1
y_train[filter1] = 1


#Build model

class_weights = {0:1.,1:((N_tot/N_rr)*1.2)}

# Fit Kernel SVM
Ncolors = np.arange(1, X_train.shape[1] + 1)


def compute_SVM(Ncolors):
    classifiers = []
    predictions = []

    for nc in Ncolors:
        # perform support vector classification
        clf = SVC(kernel='rbf', gamma=20.0, class_weight=class_weights)
        clf.fit(X_train[:, :nc], y_train)
        y_pred = clf.predict(X_test[:, :nc])

        classifiers.append(clf)
        predictions.append(y_pred)

    return classifiers, predictions

classifiers, predictions = compute_SVM(Ncolors)

completeness, contamination = completeness_contamination(predictions, y_test)


#------------------------------------------------------------
# compute the decision boundary
clf = classifiers[1]
xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 101),
                     np.linspace(ylim[0], ylim[1], 101))

Z = clf.predict(np.c_[yy.ravel(), xx.ravel()])
Z = Z.reshape(xx.shape)

# smooth the boundary
from scipy.ndimage import gaussian_filter
Z = gaussian_filter(Z, 2)

#----------------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)

# left plot: data and decision boundary
ax = fig.add_subplot(121)
im = ax.scatter(X_train[-N_plot:, 1], X_train[-N_plot:, 0], c=y_train[-N_plot:],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)

ax.contour(xx, yy, Z, [0.5], colors='k')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')

# plot completeness vs Ncolors
ax = fig.add_subplot(222)
ax.plot(Ncolors, completeness, 'o-k', ms=6)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.set_ylabel('completeness')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)

ax = fig.add_subplot(224)
ax.plot(Ncolors, contamination, 'o-k', ms=6)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))

ax.set_xlabel('N colors')
ax.set_ylabel('contamination')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)

plt.show()

#%% 







