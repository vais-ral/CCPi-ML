
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

from IPython.display import Image
import matplotlib.pyplot as plt
import scipy.io as sio

import numpy as np
import sys
import os
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.keras.metrics import categorical_accuracy

#from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
#from astroML.utils import completeness_contamination
#
##----------------------------------------------------------------------
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=False)



#Convert labels from label to CNTK output format, basically an array of 0's with a 1 in the position of the desired label so 9 = [0 0 0 0 0 0 0 0 0 1]
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        if labels[i] == 0:
            label[i,:] = np.array([0,1])
        else:
            label[i,:] = np.array([1,0])
    return label

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

#############################################################

#############Data Loading & Conversion######################
def predictionMap(xlim,ylim):

    
    mesh = []
    
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y])
            
    return (np.array(mesh))

X = np.loadtxt('AstroML_Data.txt',dtype=float)
y =  np.loadtxt('AstroML_Labels.txt',dtype=float)
print(X)
#X = X[:, [1, 0, 2, 3]]  # rearrange columns for better 1-color results
X = X[:, [1, 0]]  # rearrange columns for better 1-color results
#X = np.insert(X,X.shape[1],np.multiply(X[:,[0]],X[:,[0]]).flatten(),axis=1)
#X = np.insert(X,X.shape[1],np.multiply(X[:,[1]],X[:,[1]]).flatten(),axis=1)
X_train,y_train = reBalanceData(X,y)
filter1=y_train==0
y_train[filter1] = 0
filter1=y_train==1
y_train[filter1] = 1


#y_train = convertLabels(y_train,y_train.shape[0],2)
print(y_train.shape)
#(X_train, X_test), (y_train, y_test) = split_samples(X, y, [1, 0],
  #                                                   random_state=0)

features = X_train
#labels =np.matrix(y_train).T
labels = y_train

print('YYYY',y_train.shape)
N_tot = len(y)
#Total assignments of 0 (Classification = true)
N_st = np.sum(y == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st
N_plot = 5000 + N_rr
fig = plt.figure(figsize=(5, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(111)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(features[:, 1], features[:, 0], c=y_train,
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

###########################################################
############Netowork Building##############################
#Define input and output dimensions
#model = keras.Sequential([
#        keras.layers.Dense(2, activation=tf.nn.sigmoid),
#        keras.layers.Dense(4, activation=tf.nn.sigmoid),
#        keras.layers.Dense(2, activation=tf.nn.sigmoid),
#        keras.layers.Dense(1, activation=tf.nn.sigmoid)
#
#])
#    

model = keras.Sequential()
model.add(keras.layers.Dense(30, input_dim=2, kernel_initializer='normal', activation='sigmoid'))
model.add(keras.layers.Dense(6, input_dim=30, kernel_initializer='normal', activation='sigmoid'))

model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))

	# Compile model
#loss_fn = tf.losses.sigmoid_cross_entropy(multi_class_labels=[X_train.shape[0],2],logits=[X_train.shape[0],2])
#model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.3), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features, labels, batch_size=200,epochs=5)

3
# evaluate the model
scores = model.evaluate(features, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)
print('test',test)
predictions = np.transpose(model.predict(features))
print(predictions)


fig = plt.figure(figsize=(5, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(111)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(features[:, 1], features[:, 0], c=predictions[0],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()

test = predictionMap(xlim,ylim)
print(test)
xshape = int((xlim[1]-xlim[0])*1000)+1
yshape = int((ylim[1]-ylim[0])*1000)

predictions = np.transpose(model.predict(test[:,[1,0]]))

plt.imshow((np.reshape(predictions,(xshape,yshape))),origin='lower')
plt.colorbar()

plt.show()
print(predictions[predictions<0.2].shape,predictions[predictions>0.8].shape)



fig = plt.figure(figsize=(5, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)
ax = fig.add_subplot(111)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(test[:, 0], test[:, 1], c=predictions[0],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)
plt.show()






