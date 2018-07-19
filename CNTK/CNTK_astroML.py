# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:21:16 2018

@author: lhe39759
"""
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

from IPython.display import Image
import matplotlib.pyplot as plt
import scipy.io as sio

import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils

from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
from astroML.utils import completeness_contamination

#----------------------------------------------------------------------
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)


cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)


#CNTK Neural Network Builder
def create_modelNN(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_output_classes, activation = None)
        
        return last_layer(h)

#CNTK Neural Netowrk Builder
def create_modelLR(features):
    with C.layers.default_options(init = C.glorot_uniform()):
        r = C.layers.Dense(num_output_classes, activation = None)(features)
        return r
    
def moving_average(a, w=10):    
    if len(a) < w: 
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#Load .MAT Octave file
def loadMATData(file1):
    return sio.loadmat(file1)
   
    
# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):    
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error


#Convert labels from label to CNTK output format, basically an array of 0's with a 1 in the position of the desired label so 9 = [0 0 0 0 0 0 0 0 0 1]
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        assi = labels[i]
        if labels[i] == 10:
            assi = 0
        label[i][assi] = 1.0
    return label




#############################################################

#############Data Loading & Conversion######################

X, y = fetch_rrlyrae_combined()

filter1 = X[:,1] < 1

X[filter1]
#X = X[:, [1, 0, 2, 3]]  # rearrange columns for better 1-color results
X = X[:, [1, 0]]  # rearrange columns for better 1-color results
test = X[:10,:]
filter1 = X[:,0]>0.38
print(test[filter1,:])

X = np.insert(X,X.shape[1],np.multiply(X[:,[0]],X[:,[0]]).flatten(),axis=1)
X = np.insert(X,X.shape[1],np.multiply(X[:,[1]],X[:,[1]]).flatten(),axis=1)



(X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.3, 0.7],
                                                     random_state=0)

features = X_train
labels =np.matrix(y_train).T


###########################################################
    
##################Settings##############################

input_dim = 4
num_output_classes = 1
num_hidden_layers = 1
hidden_layers_dim = 1
mysamplesize = X_train.shape[0]
print('s',mysamplesize)






N_tot = len(y)
#Total assignments of 0 (Classification = true)
N_st = np.sum(y == 0)
print(N_tot-N_st)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st
#Number of train labels
N_train = len(y_train)
#Number of test labels
N_test = len(y_test)

N_plot = 5000 + N_rr

fig = plt.figure(figsize=(5, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)

ax = fig.add_subplot(111)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(X[:, 1], X[:, 0], c=y[:],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)


plt.show()
###########################################################
############Netowork Building##############################
#Define input and output dimensions
input1 = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

#Generate Netowrk model with CNTK layer tempate function
z = create_modelNN(input1)

loss = C.cross_entropy_with_softmax(z, label)
eval_error = C.classification_error(z, label)


#########################################################
#########Learning Parameters############################
#Alpha learning rate
learning_rate = 2000
lr_schedule = C.learning_parameter_schedule(learning_rate) 
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

minibatch_size = 59
num_samples_per_sweep = mysamplesize
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

training_progress_output_freq = 20

plotdata = {"batchsize":[], "loss":[], "error":[]}



###########################################################
#############Training########################################
for i in range(0, int(num_minibatches_to_train)):
    
    # Specify the input variables mapping in the model to actual minibatch data for training
    trainer.train_minibatch({input1 : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, 
                                                     training_progress_output_freq, verbose=0)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
        
        
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()