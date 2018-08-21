# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:57:52 2018

@author: zyv57124
"""
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle

#Load data ------------------------------------------------------
def loadMATData(file1):
    return sio.loadmat(file1)

#Load Data-------------------------------------------------------

data = loadMATData('ex3data1.mat')
features = data['X']
labels = data['y']

filter = labels ==10
labels[filter] = 0

#shuffle data---------------------------------------------------

ran = np.arange(features.shape[0])
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]

training_features = features[:3500]
training_labels = labels[:3500]
test_features = features[3501:]
test_labels = labels[3501:]

#TF Neaural Network Builder--------------------------------------

height, width, depth = 20, 20, 1
training_features = training_features.astype('float32') 
test_features = test_features.astype('float32')
test_labels = test_labels.astype('float32')

batch_size =32
num_epochs = 200
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

n_images = 5000
#
#input_shapes = np.array([20, 20, 1])
#input_shapes = input_shapes.reshape(n_images, 20, 20, 1)
print(training_features.shape)
training_features = training_features.reshape(training_features.shape[0],20, 20, 1)
print(training_features.shape)
model = Sequential()

model.add(Conv2D(batch_size, (7, 7), padding="SAME", input_shape = (20,20,1)))
model.add(Flatten())
model.add(Dense(10, activation ='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_features , training_labels, epochs=15)
test_features = test_features.reshape(test_features.shape[0],20, 20, 1)
predictions = model.predict(test_features)

count = 0
for i in range(0, len(test_labels)):
    pred = (np.argmax(predictions[i]))
    if test_labels[i][0] == pred:
        count +=1

print("Correct predictions: ", count)