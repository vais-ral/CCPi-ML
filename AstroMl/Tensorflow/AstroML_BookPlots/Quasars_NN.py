# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:40:18 2018

@author: zyv57124
"""

import numpy as np
from matplotlib import pyplot as plt
from astroML.utils import split_samples
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from astroML.classification import GMMBayes
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from astroML.utils import completeness_contamination

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

def predictionMap(xlim,ylim):
    mesh = []
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y,0,0])     
    return (np.array(mesh))
#------------------------------------------------------------
# Fetch data and split into training and test samples

quasars = np.load('quasars.npy')
stars = np.load('stars.npy')

# Truncate data for speed
quasars = quasars[::5]
stars = stars[::5]

# stack colors into matrix X
Nqso = len(quasars)
Nstars = len(stars)
X = np.empty((Nqso + Nstars, 4), dtype=float)

X[:Nqso, 0] = quasars['mag_u'] - quasars['mag_g']
X[:Nqso, 1] = quasars['mag_g'] - quasars['mag_r']
X[:Nqso, 2] = quasars['mag_r'] - quasars['mag_i']
X[:Nqso, 3] = quasars['mag_i'] - quasars['mag_z']

X[Nqso:, 0] = stars['upsf'] - stars['gpsf']
X[Nqso:, 1] = stars['gpsf'] - stars['rpsf']
X[Nqso:, 2] = stars['rpsf'] - stars['ipsf']
X[Nqso:, 3] = stars['ipsf'] - stars['zpsf']

y = np.zeros(Nqso + Nstars, dtype=int)
y[:Nqso] = 1

# split into training and test sets
(X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.9, 0.1],
                                                     random_state=0)

print (quasars.shape)
print (stars.shape)

print(X_test.shape)
#%%

LR = 0.003
Epochs = 3
BatchSize = 400
Multip = 1

N_tot = y_train.shape[0]
#Total assignments of 0 (Classification = true)
N_st = np.sum(y_train == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st
N_plot = 5000 + N_rr

#%%

model = keras.Sequential()

model.add(Dense(8, input_dim = 4, activation="tanh"))
model.add(Dense(6, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

history = model.fit(X_train, y_train, batch_size=BatchSize,epochs=Epochs, verbose=2)

predictions = np.around(model.predict(X_test).reshape(model.predict(X_test).shape[0],))

#completeness, contamination = completeness_contamination(predictions,(y_test))

scores = model.evaluate(X_test,y_test)
print(scores,model.metrics_names)
loss = scores[0]
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
#%%

def predictionMap(xlim,ylim):
    mesh = []
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y,0,0])     
    return (np.array(mesh))

def splitdata(X,y,ratio):
    length = X.shape[0]
    return X[:int(length*ratio)],X[:int(length*(1-ratio))],y[:int(length*ratio)],y[:int(length*(1-ratio))]

comp = []
cont = []
colour = []

completeness, contamination = completeness_contamination(predictions,(y_test))

scores = model.evaluate(X_test,y_test)
print(scores,model.metrics_names)
loss = scores[0]
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

comp.append(completeness)
cont.append(contamination)
color.append(coll+1)

print("completeness",completeness)
print("contamination", contamination)

loss_data = history.history['loss']
epoch_data = np.arange(0,len((loss_data)))

im_loss = ax_loss.plot(epoch_data,np.log(loss_data),'-',label=str(coll+1)+" Colours")
ax_loss.set_xlabel('Epoch')
ax_loss.legend()