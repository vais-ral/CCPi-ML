
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
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
from astroML.utils import completeness_contamination

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
            mesh.append([x,y,0,0])
            
    return (np.array(mesh))
#%%
X = np.loadtxt('AstroML_Data.txt',dtype=float)
y =  np.loadtxt('AstroML_Labels.txt',dtype=float)

print(np.amin(X[:,[2]]))
X = X[:, [1, 0]]  # rearrange columns for better 1-color results

## Shuffle Data
ran = np.arange(X.shape[0])
np.random.shuffle(ran)
X= X[ran]
y= y[ran]

X_train,y_train = reBalanceData(X,y)
#X_train, y_train = X, y

filter1=y_train==0
y_train[filter1] = 0
filter1=y_train==1
y_train[filter1] = 1

N_tot = y_train.shape[0]
#Total assignments of 0 (Classification = true)
N_st = np.sum(y_train == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st

N_plot = 5000 + N_rr

######## Plot Feature Data #########################

fig = plt.figure(figsize=(15, 15))
#Plot Size

ax = fig.add_subplot(221)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(X[:, 1], X[:, 0], c=y,
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)

ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')

#%%

###########################################################
############Netowork Building##############################

class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

model = keras.Sequential()
model.add(keras.layers.Dense(6, input_dim=2, kernel_initializer='normal', activation='sigmoid'))
model.add(keras.layers.Dense(15,  kernel_initializer='normal', activation='sigmoid'))
model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.03), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=0.3), metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=1500,epochs=100, verbose=2)


loss_data = history.history['loss']
epoch_data = np.arange(0,len(loss_data))
ax_loss = fig.add_subplot(222)
im_loss = ax_loss.plot(epoch_data,loss_data,'k-')
# evaluate the model
scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#%%
xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)
#zlim = (np.amin(X[:,[2]]),np.amax(X[:,[2]]))
#glim = (np.amin(X[:,[3]]),np.amax(X[:,[3]]))

predictions = np.transpose(model.predict(X_train))

ax_heat = fig.add_subplot(223)
im_heat = ax_heat.scatter(X_train[:, 1], X_train[:, 0], c=predictions[0], s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im_heat.set_clim(-0.5, 1)

test = predictionMap(xlim,ylim)

xshape = int((xlim[1]-xlim[0])*1000)+1
yshape = int((ylim[1]-ylim[0])*1000)

predictions =(model.predict(test[:,[1,0]]))
#%%
print(predictions.shape)
plt.imshow(np.transpose(np.reshape(predictions[:,0],(xshape,yshape))),origin='lower')
plt.colorbar()

plt.show()
print(predictions[predictions<0.2].shape,predictions[predictions>0.8].shape)

#%%

ac_cont = fig.add_subplot(224)

im_cont = ac_cont.scatter(X[:, 1],X[:, 0], c=y,s=4, lw=0, cmap=plt.cm.binary, zorder=2)
ac_cont.contour(np.reshape(test[:, 0],(xshape,yshape)), np.reshape(test[:, 1],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)),cmap=plt.cm.binary)
im.set_clim(-0.5, 1)
ac_cont.set_xlabel('$u-g$')
ac_cont.set_ylabel('$g-r$')
plt.show()




