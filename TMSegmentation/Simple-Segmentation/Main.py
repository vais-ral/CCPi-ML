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
sys.path.append(r'C:\Users\minns\OneDrive\Documents\Programming\GitHub/')
from SliceOPy import NetSlice, DataSlice
from model import u_net as unet
from keras.utils import plot_model
import tensorflow as tf
import keras
from model.losses import bce_dice_loss, bce_dice_loss_jake, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from shape_pattern_gen_segmenter import generateImage


#netData = DataSlice.NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.7,Channel_Features = (256,256), Channel_Labels = (256,256))

#unet.get_unet_256(input_shape=(256,256,1),num_classes=7)

model = NetSlice(unet.get_unet_256(input_shape=(256,256,1),num_classes=1),'keras', None)
model.loadModel(r'C:\Users\minns\OneDrive\Documents\SepNet_Project_Data\SepNet_Project_Data\Simple-Image-Segmentation-Model\modelSaveTest',customObject={'bce_dice_loss':bce_dice_loss,'dice_coeff':dice_coeff},Setting=1)
model.compileModel(keras.optimizers.RMSprop(lr=0.001), bce_dice_loss, [dice_coeff])
#model.generativeDataTrain(generateImage, BatchSize=1, Epochs=100,Channel_Ordering=(256,256,1,1))
#model.trainModel(Epochs = 1,Batch_size = None, Verbose = 2)

#model.plotLearningCurve()
#print(model.getHistory())

#%%
#model.saveModel('modelSaveTest')
#%%
#features = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\noisy_rect_ellip.npy')
#labels = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\rect_ellip.npy')
#
#netData = NetData(features,labels,Shuffle=True,Rebalance=None,Split_Ratio = 0.1)
##%%

model.generativeDataTesting(generateImage, SampleNumber=1,Threshold=1e-3,Channel_Ordering=(256,256,1,1))
#data = []
#item = generateImage()
#data.append(np.array(item))
#data = np.array(data)
#feat , labels,shape = model.channelOrderingFormat(np.array(data[0][0]), np.array(data[0][1]),256,256)
#netData = DataSlice.DataSlice(Features =feat,Labels= labels,Channel_Features =(256,256), Channel_Labels = (256,256), Split_Ratio = 1.0)
#print(netData.X_test)
#predictions = model.predictModel(feat)
###%%
#print(np.array(predictions).shape)
#fig = plt.figure(figsize=(15, 15))
#fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
#ax_loss = fig.add_subplot(121)
#ax_loss.imshow(feat[0].reshape(256,256))
#ax_loss = fig.add_subplot(122)
#ax_loss.imshow(np.array(predictions[0]).reshape(256,256))
#plt.show()
####%%