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


###########################################################
    
##################Settings##############################

input_dim = 400
num_output_classes = 10
num_hidden_layers = 1
hidden_layers_dim = 25
mysamplesize = 5000

#############################################################

#############Data Loading & Conversion######################

data = loadMATData('ex3data1.mat')
features = data['X']
labels = data['y']

#shuffle data
ran = np.arange(features.shape[0])
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]
print("Feat",features.shape)
#Convert to CNTK format
labels = convertLabels(labels,mysamplesize,num_output_classes)
print("Label",labels.shape)

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
learning_rate = 0.02
lr_schedule = C.learning_parameter_schedule(learning_rate) 
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

minibatch_size = 64
num_samples_per_sweep = 5000
num_sweeps_to_train_with = 100
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