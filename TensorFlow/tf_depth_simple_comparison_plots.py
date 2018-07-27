# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:24:50 2018

@author: zyv57124
"""

import numpy as np
import matplotlib.pyplot as plt

for a in np.arange(1, 500, 10):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('loss_data_batchnum_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(epoch, loss)

plt.title('Loss against Epoch For Changing Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

for a in np.arange(1, 500, 10):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('loss_data_batchnum_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(batch_size, delta_loss)
    
plt.title('Change in Loss against Batch Size For Changing Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Change in Loss')
plt.show()

for a in np.arange(1, 500, 10):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('loss_data_batchnum_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(epoch, delta_loss)
    
plt.title('Change in Loss against Epoch For Changing Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Change in Loss')
plt.show()

for a in np.arange(1, 500, 10):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('loss_data_batchnum_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(batch_size, time)
    
plt.title('Training Time against Batch Size For Changing Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Training Time')
plt.show()

for a in np.arange(1, 500, 10):
    epoch , loss, batch_size, time, delta_loss = np.genfromtxt('loss_data_batchnum_' + str(a) +'.txt', delimiter=',', unpack=True)
    plt.plot(epoch, delta_loss)
    
plt.title('Change in Loss Between Epochs against Epoch Number')
plt.xlabel('Epoch')
plt.ylabel('Change in Loss Between Epochs')
plt.show()

