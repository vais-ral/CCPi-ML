# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:29:48 2018

@author: lhe39759
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot

def plotGaussian(labels,xmin,xmax,ymin,ymax,spacerx,spacery,label):
    x =np.arange(xmin,xmax, (np.abs(xmin)+np.abs(xmax))/spacerx)
    y = np.arange(ymin, ymax, (np.abs(ymin)+np.abs(ymax))/spacery)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gaus = labels.reshape(X.shape)
    ax.plot_surface(X, Y, gaus)
    fig.suptitle(label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')   
    plt.show()
    
def lossPlot(loss,label):
    
    epoch = np.arange(0, len(loss))
    plt.plot(epoch,loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

layer1Neurons = [1,2,3,4,5,7,9,12,15,20,30,40,50]
 
layer2Neurons = [0,1,2,3,4,5,7,9,12,15]

layer1Neurons = [1,2,3,4,5,6,7,9,12,15,20,30,40,50]
layer2Neurons =  [0,1,2,3,4,5,6,7,9,12,15,20,30,40,50]

l1 = (len(layer1Neurons))
l2 = (len(layer2Neurons))

surface = np.load(r'C:\Users\lhe39759\Documents\CCPI-Data\Gaussian-Hill-Final-Grid-Data/surface.npy')
losses = np.load(r'C:\Users\lhe39759\Documents\CCPI-Data\Gaussian-Hill-Final-Grid-Data/history.npy')
params = np.load(r'C:\Users\lhe39759\Documents\CCPI-Data\Gaussian-Hill-Final-Grid-Data/params.npy')
#%%
loss = []

for l in losses:
    loss.append(np.array(l['loss']))
loss = np.array(loss)
val_loss = []

for l in losses:
    val_loss.append(np.array(l['val_loss']))
val_loss = (np.array(val_loss))

print(params.shape)
print("ddd",loss.shape,(val_loss).shape)
print(surface.shape)

params = params.reshape(l2,l1)
loss = loss.reshape(l2,l1,loss.shape[1])
val_loss = np.transpose(val_loss.reshape(l2,l1,val_loss.shape[1]),axes=(1,0,2))

surface = surface.reshape(l2,l1,surface.shape[1],surface.shape[2])

params = np.transpose(params,axes = (1,0))
loss = np.transpose(loss,axes = (1,0,2))
surface = np.transpose(surface,axes = (1,0,2,3))


print(params.shape)
print(loss.shape,val_loss.shape)
print(surface.shape)
#%%
print(params)
lossgrid = loss[:,:5,-1]
print(lossgrid.shape)
#%%
maxi = np.amax(lossgrid)
#
max
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log(lossgrid/maxi), aspect='auto')
#plt.imshow(lossgrid, aspect='auto')
print(params.shape)
print(val_loss.shape,loss.shape)
print(maxi)
for i in range(params.shape[0]):
    c=0
    if i == 0:
        c = 1
    for j in range(5):
        text = ax.text(j, i, str(params[i, j])+" ("+str("%.9f" %  np.log(loss[i,j,-1]))+")", ha="center", va="center", color="w",weight='heavy',size='xx-large')
ax.set_xlabel("2nd Hidden Layer Width")
ax.set_ylabel("1st Hidden Layer Width")

a = 1
b = 3
print(val_loss.shape)
#ax = fig.add_subplot(122)
#
#ax.plot(np.arange(0,loss[a][b].shape[0]),(loss[a][b]))
plt.show()
#plotGaussian(surface[13][13],-5.0,5.0,-5.0,5.0,100,100,"Hill Valley")

#lossPlot(loss[0],"loss")
#plt.show()
#
#dets = np.load('netDetails.npy')
#params = np.load('params.npy')
#
#print(dets[0],params[0])