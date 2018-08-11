# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:16:19 2018

@author: lhe39759
"""
import matplotlib.pyplot as plt

import numpy as np
from Network import NetModel
from NetData import NetData
from model import u_net as unet
from keras.utils import plot_model
import tensorflow as tf
import keras
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff


net = {"Input":(4),
       
       "HiddenLayers":[
            {"Type": 1,
            "Width": 4,
            "Activation": "tanh"
            },
            {"Type": 1,
            "Width": 3,
            "Activation": "tanh"
            },
            {"Type": 1,
            "Width": 1,
            "Activation": "sigmoid"
            }]

       }

layers = []
layers.append(keras.layers.Dense(4,input_dim=(4),kernel_initializer='normal', activation="tanh"))
layers.append(keras.layers.Dense(3,kernel_initializer='normal', activation="tanh"))
layers.append(keras.layers.Dense(1,kernel_initializer='normal', activation="sigmoid"))

modelNet = keras.Sequential(layers)



features = np.load(r'/home/jake/Documents/Programming/Python/CCPi-ML/TensorFlow/AstroMl/AstroML_Data/AstroML_Data_shuffled.npy')[:,[1,0,2,3]]
labels = np.load(r'/home/jake/Documents/Programming/Python/CCPi-ML/TensorFlow/AstroMl/AstroML_Data/AstroML_Labels_shuffled.npy')
print(features.shape)
netData = NetData(features,labels,Shuffle=True,Rebalance=0.1,Split_Ratio = 0.7)

#unet.get_unet_256(input_shape=(256,256,1))

model = NetModel(modelNet,'keras', netData)

#model.loadModel('modelSave.h5',{ 'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff })


#model.compileModel(keras.optimizers.RMSprop(lr=0.0001),'binary_crossentropy', ['binary_accuracy', 'categorical_accuracy'])

model.compileModel(keras.optimizers.Adam(lr=0.001),'binary_crossentropy', ['binary_accuracy', 'categorical_accuracy'])
model.trainModel(Epochs = 10,Batch_size = 100, Verbose = 2)

model.plotLearningCurve()

#%%
#model.saveModel('modelSave.h5')
#%%
#features = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\noisy_rect_ellip.npy')
#labels = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\rect_ellip.npy')
#
#netData = NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.1)
#%%
#predictions = model.predict(netData.X_train)
##%%
#fig = plt.figure(figsize=(15, 15))
#fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
#ax_loss = fig.add_subplot(121)
#ax_loss.imshow(netData.X_train[0].reshape(256,256))
#ax_loss = fig.add_subplot(122)
#ax_loss.imshow((predictions[0].reshape(256,256)))
#plt.show()
#%%