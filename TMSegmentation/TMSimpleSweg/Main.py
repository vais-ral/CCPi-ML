# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:16:19 2018

@author: lhe39759
"""
import numpy as np
from Network import NetModel
from NetData import NetData
from model import u_net as unet
from keras.utils import plot_model
import tensorflow as tf
import keras
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

"""
net = {"Input":(4),
       
       "HiddenLayers":[
            {"Type": 1,
            "Width": 25,
            "Activation": "tanh"
            },
            {"Type": 1,
            "Width": 16,
            "Activation": "sigmoid"
            }]

       }
"""



features = np.load(r'/home/jake/Documents/Programming/Python/CCPi-ML/TensorFlow/AstroMl/AstroML_Data/AstroML_Data_shuffled.npy')[:10]
labels = np.load(r'/home/jake/Documents/Programming/Python/CCPi-ML/TensorFlow/AstroMl/AstroML_Data/AstroML_Labels_shuffled.npy')[:10]

netData = NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.1)

#netData.channelOrderingFormatTest(256,256)
#netData.channelOrderingFormatTrain(256,256)
print(netData.X_train.shape,netData.X_test.shape)
model = NetModel(unet.get_unet_256(input_shape=(256,256,1)),'keras')
model.compileModel(keras.optimizers.RMSprop(lr=0.0001),bce_dice_loss, [dice_coeff])
model.trainModel(netData.X_train, netData.y_train, Epochs = 10,Batch_size = 1, Verbose = 2)
print(model.summary())