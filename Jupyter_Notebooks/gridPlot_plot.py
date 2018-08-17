# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:29:48 2018

@author: lhe39759
"""

import numpy as np
import matplotlib.pyplot as plt

def lossPlot(loss,label):
    
    epoch = np.arange(0, len(loss))
    plt.plot(epoch,loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
surface = np.load('surface.npy')

plotGaussian(surface[0],-5.0,5.0,-5.0,5.0,100,200,"Hill Valley")

loss = np.load('history.npy')
print(loss[0])
#lossPlot(loss[0],"loss")
#plt.show()
#
#dets = np.load('netDetails.npy')
#params = np.load('params.npy')
#
#print(dets[0],params[0])