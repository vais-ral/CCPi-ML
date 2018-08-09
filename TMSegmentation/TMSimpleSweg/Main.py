# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:16:19 2018

@author: lhe39759
"""
import numpy as np
from Network import NetModel

from NetData import *
from model import u_net as unet
from keras.utils import plot_model
import tensorflow as tf
import keras

def loadTrainingData(fileF,fileL):
    
    return np.load(fileF),np.load(fileL)

#net = {"Input":(100,100,1),
#       
#       "HiddenLayers":[
#           {"Type": 0,
#            "Width": 36,
#            "Activation": "relu",
#            "Kernel": (3,3)
#            },
#           {"Type": 3,
#            "Kernel": (3,3)
#            },
#           {"Type": 0,
#            "Width": 24,
#            "Activation": "relu",
#            "Kernel": (3,3)
#            },
#           {"Type":4
#            },
#            {"Type": 1,
#            "Width": 20000,
#            "Activation": "relu"
#            },
#            {"Type": 1,
#            "Width": 10000,
#            "Activation": "sigmoid"
#            }]
#
#       }
#
#
#
#model = NetModel(net,'keras')
##plot_model(model, to_file='model.png')
def channelOrderingFormat(Feat_train,Feat_test,img_rows,img_cols):
    if keras.backend.image_data_format() == 'channels_first':
        Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
        Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
        Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)  
    return Feat_train, Feat_test, input_shape

features = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\nosey_images4.npy')[:10]
labels = np.load(r'C:\Users\lhe39759\Documents\GitHub\Data_TM\RectangeSimple\images4.npy')[:10]

X_train,X_test,y_train,y_test = splitData(features,labels,0.7)
print(np.amax(y_train[0]))
X_train,X_test,image_shape = channelOrderingFormat(X_train,X_test,256,256)
y_train,y_test,image_shape2 = channelOrderingFormat(y_train,y_test,256,256)

model = unet.get_unet_256(input_shape=(256,256,1))
print(model.summary())

#%%
print(X_train.shape)
#%%
model.compile(optimizer=keras.optimizers.Adam(lr=0.000001), loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, batch_size=1,epochs=10, verbose=2)
print(model.summary())