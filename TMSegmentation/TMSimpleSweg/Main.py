# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:16:19 2018
#       "HiddenLayers":[
#            {"Type": 1,
#            "Width": 4,
#            "Activation": "tanh"
#            },
#            {"Type": 1,
#            "Width": 3,
#            "Activation": "tanh"
#            },
#            {"Type": 1,
#            "Width": 1,
#            "Activation": "sigmoid"
#            }]
#
#       }
#


@author: lhe39759
"""
import matplotlib.pyplot as plt

import numpy as np
import sys
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub/')
from SliceOPy import NetSlice, DataSlice
from model import u_net as unet
from keras.utils import plot_model
import tensorflow as tf
import keras
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from random_shapes_gen import generateImage


#netData = DataSlice.NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.7,Channel_Features = (256,256), Channel_Labels = (256,256))

#unet.get_unet_256(input_shape=(256,256,1))

model = NetSlice.NetSlice(unet.get_unet_256(input_shape=(256,256,1)),'keras', None)
model.compileModel(keras.optimizers.RMSprop(lr=0.001), bce_dice_loss, [dice_coeff])
model.generativeDataTrain(generateImage, BatchSize=1, Epochs=500)
#model.trainModel(Epochs = 10,Batch_size = None, Verbose = 2)

model.plotLearningCurve()

#%%
model.saveModel('modelSaveTest')
#%%
#features = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\noisy_rect_ellip.npy')
#labels = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\rect_ellip.npy')
#
#netData = NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.1)
##%%
#predictions = model.predictModel(netData.X_test[:2])
##%%
#fig = plt.figure(figsize=(15, 15))
#fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
#ax_loss = fig.add_subplot(121)
#ax_loss.imshow(netData.X_train[1].reshape(256,256))
#ax_loss = fig.add_subplot(122)
#ax_loss.imshow((predictions[1].reshape(256,256)))
#plt.show()
##%%