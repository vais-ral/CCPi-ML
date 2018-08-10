# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:19:08 2018

@author: lhe39759
"""
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from NetData import NetData
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

class NetModel:

#""" Model Building
#    Choice of backends are: 'keras' 
#    
#    Network input should have structure of
#    
#    {"Input": tuple for convolution input (width,height,channels), int for data,
#    "HiddenLayers": List of hiddenlayers see below for structure, avalible options are:
#        Convolution ->
#        
#         {"Type": 'Conv2D',
#            "Width": Layer Width, int E.G 10
#            "Activation": Activation type E.G "relu","tanh","sigmoid",
#            "Kernel": Kernal Size, tuple E.G (3,3)
#            }
#        Dense ->
#        
#         {"Type": 'Dense',
#            "Width": Layer Width, int E.G 10,
#            "Activation":  Activation type E.G "relu","tanh","sigmoid""
#            }
#         
#        Dropout ->
#            {"Type": 'Dropout,
#            "Ratio": Ratio of neurons to drop out, float E.G 0.7
#            }
#        
#        Pooling -> 
#        {"Type": 'Pooling',
#        "kernal": Kernal Size, tuple E.G (3,3)}
#        "dim": dimensions of dropout, int E.G 1,2,3
#
#        } 
#   
#       Flatten - >
#        {"Type": 'Flatten,
#        } 
#    }    
#
#    """
#CANT HAVE FLATTERN AS FIRST LAYER YET    
    
    def __init__(self,Network,Backend):
        
        self.backend = Backend
        self.history = None
        self.netData = None
        if type(Network) == dict:
            self.input = Network["Input"]
            self.hiddenLayers = Network["HiddenLayers"]
            self.model = self.buildModel()
        elif (Network) == None:
            print("Empty Network Created,use model.loadModel(path,custom_object) function to load model.")
        else:# type(Network) == type(keras.engine.training.Model):
            self.model = Network
    
    def loadData(self,netDataObj):
        self.netData = netDataObj  
    
    def buildModel(self):
        if self.backend == 'keras':
            return self.kerasModelBackend()
 
    def loadModel(self,name,customObject):
        if self.backend == 'keras':
            if customObject is not None:
                self.model = keras.models.load_model(name,custom_objects=customObject)
            else:
                self.model = keras.models.load_model(name)                
        
    def saveModel(self,name):
        if self.backend == 'keras':
            self.model.save(name)

    def compileModel(self,Optimizer, Loss, Metrics):
        if self.backend== 'keras':
            self.kerasCompileModel(Optimizer,Loss,Metrics)
                
    def trainModel(self,Epochs = 100,Batch_size =1, Verbose = 2):
        if self.netData != None:
            if self.backend== 'keras':
                self.history = self.kerasTrainModel(Epochs,Batch_size,Verbose)
        else:
            print("Please load data into model first using model.loadData(NetData)")
        
        
    def predictModel(self,testData):
        if self.backend== 'keras':
            return self.kerasPredictModel(testData)

    def summary(self):
        if self.backend== 'keras':
            return self.model.summary()
    
    def gpuCheck(self):
        if self.backend== 'keras':
            keras.backend.tensorflow_backend._get_available_gpus()
            
##################################       
        
### Keras Backend ############

##################################

    def kerasModelBackend(self):
            layers = []
            #Input layer
            print('num',len(self.hiddenLayers))

            dataModifier = False

            for layer in range(0,len(self.hiddenLayers)):    
                print('type',self.hiddenLayers[layer]["Type"])
                #Check for first layer to deal with input shape
                
                if layer == 0 or dataModifier:
                    #Convolution first layer
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], input_shape=(self.input),padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                        dataModifier = False
                    #Dense first layer
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],input_dim=(self.input),kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))
                        dataModifier = False
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
                        dataModifier = True

                else:
                # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flatten 
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))       
                    elif self.hiddenLayers[layer]["Type"] == "Dropout":
                        layers.append(keras.layers.Dropout(self.hiddenLayers[layer]["Ratio"]))    
                    elif self.hiddenLayers[layer]["Type"] == "Pool":
                        layers.append(keras.layers.MaxPooling2D(pool_size=self.hiddenLayers[layer]["Kernal"],data_format= keras.backend.image_data_format()))
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
            return keras.Sequential(layers)               

            

    def kerasCompileModel(self,Optimizer,Loss,Metrics):
        self.model.compile(optimizer=Optimizer, loss=Loss, metrics=Metrics)
        
    def kerasTrainModel(self,Epochs,BatchSize,Verbose):
        return self.model.fit(self.netData.X_train, self.netData.y_train, validation_data=(self.netData.X_test,self.netData.y_test), batch_size=BatchSize,epochs=Epochs, verbose=Verbose)

    def kerasPrecictModel(self,testData):
        return self.model.predict(testData)
           

    














































#################################################################

## Pytorch Backend

###############################################################
"""
    def pyTorchModelBackend(self):
            layers = []
            activaions  = {"relu":torch.nn.Relu(True),"tanh":torch.nn.Tanh(True),"sigmoid":torch.nn.Sigmoid(True)}
            #Input layer
            print('num',len(self.hiddenLayers))
            for layer in range(0,len(self.hiddenLayers)):    
                print('type',self.hiddenLayers[layer]["Type"])
                #Check for first layer to deal with input shape
                
                if layer == 0:
                    #Convolution first layer
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(torch.nn.Conv2D(self.input[2], self.hiddenLayers[layer]["Width"], kernal_size=(self.hiddenLayers["Kernal"])))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])

                    #Dense first layer
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(torch.nn.Linear(self.input,self.hiddenLayers[layer]["Width"]))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                else:
                # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flatten 
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(torch.nn.Conv2D(self.input[2], self.hiddenLayers[layer]["Width"], kernal_size=(self.hiddenLayers["Kernal"])))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(torch.nn.Linear(self.input,self.hiddenLayers[layer]["Width"]))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                    elif self.hiddenLayers[layer]["Type"] == "Dropout":
                        
                        if self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.Dropout(p=self.hiddenLayers[layer]["Ratio"]))
                        elif self.hiddenLayers[layer]["Dim"] == 2:
                            layers.append(torch.nn.Dropout2d(p=self.hiddenLayers[layer]["Ratio"]))
                        elif self.hiddenLayers[layer]["Dim"] == 3:
                            layers.append(torch.nn.Dropout3d(p=self.hiddenLayers[layer]["Ratio"]))
                    
                    elif self.hiddenLayers[layer]["Type"] == "Pool":
                        if self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool1d(self.hiddenLayers[layer]["Kernal"]))
                        elif self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool2d(self.hiddenLayers[layer]["Kernal"]))
                        elif self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool3d(self.hiddenLayers[layer]["Kernal"]))
                    
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
            return torch.nn.Sequential(*layers)
"""
#Save Model
#def saveModel(model,name):
#    
#    save = "d"
#    
#    while(save != "y" or save != "n"):
#
#        save = input("Do you want to save the model (y/n)?")
#    
#        if save == "y":
#            model.save(name)
#            break
#        elif save == "n":
#            break
#
##Generate new model or save model
#def loadOrGenModel(input_shape,name):
#   
#    load = "d"
#    
#    while(load != "y" or load != "n"):
#
#        load = input("Load existing model (y/n)?")
#    
#        if load == "y":
#            return model.loadModel(name)
#        elif load == "n":  
#            return buildModel(input_shape,1)
#          

#Add check for input shape channel thing
        


