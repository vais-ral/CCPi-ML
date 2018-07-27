# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:21:16 2018

@author: lhe39759
"""
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

from IPython.display import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import numpy as np
import cntk as C
import cntk.tests.test_utils
import time
    
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)

##--------------------------------------------------------------------------------------------



##--------------------------------------------------------------------------------------------
    
    
#CNTK Neural Network Builder
def create_modelNN(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.tanh):
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

    training_loss = trainer.previous_minibatch_loss_average
    eval_error = trainer.previous_minibatch_evaluation_average
    if verbose: 
        print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

def generate_minibatch(features, labels,batch, batchsize):

     start = batch*batchsize 
     return features[start:start+batchsize],labels[start:start+batchsize]
 
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
#Convert to CNTK format
labels = convertLabels(labels,mysamplesize,num_output_classes)

features = features[:3500]
labels = labels[:3500]
###########################################################
############Netowork Building##############################
for bb in np.arange(0,50,4):

    if bb == 0:
        bb = 1
        
    num_hidden_layers = bb
    hidden_layers_dim = 4
 
    print('batch',bb)
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
    learning_rate = 0.1
    lr_schedule = C.learning_parameter_schedule(learning_rate) 
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, eval_error), [learner])
    
    minibatch_size =(40)
    num_samples_per_sweep = 3500.0
    num_sweeps_to_train_with = 1.0
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / float(minibatch_size)
    
    training_progress_output_freq = 1
    
    plotdata = {"epoch":[],"batch":[], "loss":[], "deltaloss":[],"speed":[]}
    
    
    limit = 200
    
        
    
    ###########################################################
    #############Training########################################
    
    for epoch in range(0,limit):
        epochLoss = 0
        start = time.time()

        for batch in range(0, int(math.ceil(num_minibatches_to_train))+1):
            if(batch*minibatch_size>=features.shape[0]):
                break
            train_features, train_labels = generate_minibatch(features,labels,batch, minibatch_size)
            # Specify the input variables mapping in the model to actual minibatch data for training
            trainer.train_minibatch({input1 :  train_features, label : train_labels})
            batchsize, runningloss, error = print_training_progress(trainer, epoch, 
                                                             training_progress_output_freq, verbose=0)
            epochLoss += runningloss
        

        epochLoss = epochLoss/(int(math.ceil(num_minibatches_to_train)))

        end = time.time()
        if True:
            epoch+=1
            if not (epochLoss == "NA" or error =="NA"):
                plotdata["epoch"].append(epoch)
                plotdata["batch"].append(bb)
                if epoch == 1:
                    plotdata["deltaloss"].append('Nan')
                else:
                    plotdata["deltaloss"].append(float(plotdata["loss"][-1])-epochLoss)
                plotdata["loss"].append(epochLoss)
                plotdata["speed"].append(end-start)
    
    
            #plotdata["error"].append(error)
            
    f = open('Data\VDepth_3\cntk_data_batchnum_'+str(bb)+".txt","w")
    
    for i in range(0,len(plotdata["epoch"])):
        f.write(str(plotdata["epoch"][i])+","+str(plotdata["batch"][i])+","+str(plotdata["loss"][i])+","+str(plotdata["deltaloss"][i])+","+str(plotdata["speed"][i])+"\n")
        
    f.close()
       
#plotdata["avgloss"] = moving_average(plotdata["loss"])
#plotdata["avgerror"] = moving_average(plotdata["deltaloss"])
#
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["epoch"], plotdata["loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
#
#plt.subplot(212)
#plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
#plt.xlabel('Minibatch number')
#plt.ylabel('Label Prediction Error')
#plt.title('Minibatch run vs. Label Prediction Error')
#plt.show()