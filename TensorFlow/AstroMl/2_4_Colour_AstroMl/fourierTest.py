# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:09:59 2018

@author: lhe39759
"""
import tensorflow as tf
import keras
from scipy.fftpack import fft, ifft, fftn, ifftn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification,make_swiss_roll,make_sparse_uncorrelated

def func(x,y):
    
    y1 = ((((x-0.5)**2.0)-0.25)**0.5)+0.5
    print(x,y1)
    if y <= y1:
    
        return 0
    else:
        return 1
    
def lin(x,y):
    y1 = x
    if y<=y1:
        return 0
    
    else:
        return 1

def predictionMap(xlim,ylim):
    
    mesh = []
    for x in np.arange(xlim[0],xlim[1],0.01):
        for y in np.arange(ylim[0],ylim[1],0.01):
            mesh.append([x,y])     
    return (np.array(mesh))

plt.clf()
fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
ax_loss = fig.add_subplot(131)

X_train = np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy')
y_train = np.load('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy')

origData = ( make_circles(n_samples=10000,noise=0.1, random_state=0))
data = np.append(np.array(origData[0]),origData[1].reshape(origData[1].shape[0],1),axis=1)

ax_loss = fig.add_subplot(121)
data[:,[0]] = data[:,[0]] +1.5
data[:,[1]] = data[:,[1]] +1.5
ax_loss.scatter(data[:,[0]],data[:,[1]],c=data[:,[2]])

yf = (fft((data)))
print(yf[:,[0,1,2]])
#yf[:,[0]] = yf[:,[0]] /3.0
#yf[:,[1]] = yf[:,[1]] /6.0
model = keras.Sequential()

model.add(keras.layers.Dense(4, input_dim= 2,activation='tanh'))
model.add(keras.layers.Dense(3,activation='tanh'))

#    model.add(keras.layers.Dense(64))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1,activation='sigmoid'))
    
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

history = model.fit(np.abs(yf[:,[0,1]]),data[:,[2]], batch_size=100,epochs=10, verbose=2)


xlim = (0.0, 6.0)
ylim = (0.0,3.0)
		
		#predictions = np.transpose(model.predict(X_train))

test = predictionMap(xlim,ylim)
print(test[0:10])

predictions =(model.predict(test[:,[0,1]]))

predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
ax_loss = fig.add_subplot(132)

xshape = int((xlim[1]-xlim[0])*100)
yshape = int((ylim[1]-ylim[0])*100)
print(xshape,yshape)
ax_loss.scatter((test[:,[0]]),(test[:,[1]]), c=predictions)
cb = fig.colorbar(im_heat, ax=ax_loss)

ax_loss.scatter(np.abs(yf[:,[0]]),np.abs(yf[:,[1]]), c=data[:,[2]])

#ax_loss = fig.add_subplot(133)
print(np.abs(yf[:,[0,1,2]])[0:10])

iTest = np.append(test,predictions,axis=1)
print(iTest[0:10])

iTest = ifft(iTest)
print(np.abs(iTest)[0:10])



ax_loss = fig.add_subplot(133)

#%%
yf = ifft(yf)
ax_loss.scatter(np.abs(yf[:,[0]]),np.abs(yf[:,[1]]), c=data[:,[2]])

ax_loss.scatter(np.abs(iTest[:,[0]]),np.abs(iTest[:,[1]]), c=predictions)
plt.show()
