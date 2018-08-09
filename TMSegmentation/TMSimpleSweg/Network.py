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

        if type(Network) == dict:
            self.input = Network["Input"]
            self.hiddenLayers = Network["HiddenLayers"]
            self.model = self.buildModel()

        else:# type(Network) == type(keras.engine.training.Model):
            print("MODEL")
            self.model = Network
        
    def buildModel(self):
        if self.backend == 'keras':
            return self.kerasModelBackend()
 
    def loadModel(self,name):
        if self.backend == 'keras':
            return keras.models.load_model(name)
        
    def saveModel(self,name):
        if self.backend == 'keras':
            self.model.save(name)

    def compileModel(self,Optimizer, Loss, Metrics):
        if self.backend== 'keras':
            self.kerasCompileModel(Optimizer,Loss,Metrics)
                
    def trainModel(self,features,labels,Epochs = 100,Batch_size =1, Verbose = 2):
        if self.backend== 'keras':
            return self.kerasTrainModel(features,labels,Epochs,Batch_size,Verbose)    
        
    def channelOrderingFormat(self,Feat_train,Feat_test,img_rows,img_cols):
        if self.backend== 'keras':
            return self.kerasChannelOrderingFormat(Feat_train,Feat_test,img_rows,img_cols)

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
        
    def kerasTrainModel(self,features,labels,Epochs,BatchSize,Verbose):    
        return self.model.fit(features, labels, batch_size=BatchSize,epochs=Epochs, verbose=Verbose)
      
           

    














































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
        


