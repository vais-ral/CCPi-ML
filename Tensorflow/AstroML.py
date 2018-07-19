# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.io as sio
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadMATData(file1):
    return sio.loadmat(file1)

def classificaionContour(model,xmin,xmax,ymin,ymax,step):
		contour = np.zeros((xmax-xmin)*(ymax-ymin))
		print(contour.shape)
		counter = 0
		for x in np.arange(xmin,xmax,1):
			for y in np.arange(ymin,ymax,1):
				ar = np.array([[np.divide(x,100.0),np.divide(y,100.0)]])
				print(ar,model.predict(ar))
				contour[counter] =model.predict(ar)
				counter+=1
		return np.transpose(np.reshape(contour,(xmax-xmin,ymax-ymin)))
    
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):   
        label[i][0] = labels[i]
    return label

features = np.loadtxt('AstroML_Data.txt',dtype=float)
labels =  np.loadtxt('AstroML_Labels.txt',dtype=float)
features = features[:, [1, 0]]
#print(features)
filter1 = labels == 1
filtered_lab = np.extract(filter1,labels)
filter2 = np.array([filter1,filter1])
filter2 = np.transpose(filter2)
print(filter2)
feat_trans =np.transpose(features)
comp1 = feat_trans[0]
comp2 = feat_trans[1]
comp1 = np.extract(filter1,comp1)
comp2 = np.extract(filter1,comp2)

feat = np.array([comp1,comp2])
feat = np.transpose(feat)
filterd_features = feat

print(filterd_features.shape,filtered_lab.shape)



for i in range(0,192):
    features = np.append(features,filterd_features,axis=0)

    labels = np.append(labels,filtered_lab,axis=0)




ran = np.arange(labels.shape[0])
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]
print("Feat",features.shape,labels.shape)

filter1 = labels == 1
labels[filter1] = 0.99

#labels = convertLabels(labels,len(labels),1)

#feat_train = features[:3500]
#labels_train = labels[:3500]
#feat_test = features[3501:]
#labels_test = labels[3501:]


print(np.sum(labels == 0.99),np.sum(labels==0))

print(features)

model = keras.Sequential([
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(features, labels, epochs=15)



xlim = (70, 135)
ylim = (-15, 40)

#Produce probability from previously generated 2-D array based on previous classification to be plotted as heatmap 

#Reshape the resulting prediction the same as xx 2-D array
Z = classificaionContour(model,xlim[0],xlim[1],ylim[0],ylim[1],1)
print(Z)
plt.imshow(Z)
print(np.sum(labels > 0.5),np.sum(labels==0))


#plt.set_clim(0, 1.5)
#plot decision bounda

plt.show()