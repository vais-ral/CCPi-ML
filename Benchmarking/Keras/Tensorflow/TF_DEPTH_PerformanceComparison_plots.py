# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:00:20 2018

@author: zyv57124
"""
import numpy as np
import matplotlib.pyplot as plt

for a in np.arange(1, 30, 1):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('TF_changing_depth_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(epoch, loss)

plt.title('Loss against Epoch For Changing Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



for a in np.arange(1, 30, 1):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('TF_changing_depth_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(epoch, delta_loss)
    
plt.title('Change in Loss Between Epochs against Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Change in Loss Between Epochs')
plt.show()

plt.title('Change in Loss against Batch Size For Changing Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Change in Loss')
plt.show()


for a in np.arange(1, 30, 1):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('TF_changing_depth_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(batch_size, time)
    
plt.title('Change in Loss Between Epochs against Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Change in Loss Between Epochs')
plt.show()

plt.title('Change in Loss against Batch Size For Changing Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Change in Loss')
plt.show()
