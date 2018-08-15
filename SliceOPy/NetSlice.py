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
from SliceOPy import DataSlice#from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

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
    
    def __init__(self,Network,Backend,netData = None):
        
        self.backend = Backend
        self.history = None
        self.netData = None

        #three types of model initilisation, from custom dictionary object, Empty Model, Direct Model Input
        if type(Network) == dict:
            self.input = Network["Input"]
            self.hiddenLayers = Network["HiddenLayers"]
            self.model = self.buildModel()
        elif (Network) == None:
            print("Empty Network Created,use model.loadModel(path,custom_object) function to load model.")
        else:# type(Network) == type(keras.engine.training.Model):
            self.model = Network

        if netData!=None:
            self.loadData(netData)
            
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
                
    def trainModel(self,Epochs = 100,Batch_size =None, Verbose = 2):
        if self.netData != None:
            #If batch size is none set batch size ot size of training dataset
            if Batch_size is None:
                Batch_size = self.netData.X_train.shape[0]
            # Training should return dictionary of loss ['loss'] and cross validation loss ['val_loss'] 
            if self.backend== 'keras':
                self.history = self.kerasTrainModel(Epochs,Batch_size,Verbose)
        else:
            print("Please load data into model first using model.loadData(NetData)")
        
        
    def predictModel(self,testData):
        if self.backend== 'keras':
            return self.kerasPrecictModel(testData)

    def summary(self):
        if self.backend== 'keras':
            return self.model.summary()
    
    def gpuCheck(self):
        if self.backend== 'keras':
            keras.backend.tensorflow_backend._get_available_gpus()
    
    def getHistory(self):
        return self.history

    def plotLearningCurve(self):

        loss = []
        val_loss = []

        if self.backend== 'keras':
            loss,val_loss = self.kerasGetHistory()

        epochs = np.arange(0,len(loss),1)

        plt.plot(epochs,loss,label="Training Data")
        plt.plot(epochs,val_loss,label="Validation Data")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()


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
        print(self.netData.X_train.shape,self.netData.X_test.shape, self.netData.y_train.shape,self.netData.y_test.shape)
        return self.model.fit(self.netData.X_train, self.netData.y_train, validation_data=(self.netData.X_test,self.netData.y_test), batch_size=BatchSize,epochs=Epochs, verbose=Verbose)

    def kerasPrecictModel(self,testData):
        return self.model.predict(testData)
           
    def kerasGetHistory(self):
        return self.history.history['loss'],self.history.history['val_loss']
    
    def contourPlot(self):
        
        x1_min_tr = np.amin(self.netData.X_train[:,0])
        x1_max_tr = np.amax(self.netData.X_train[:,0])
        x2_min_tr = np.amin(self.netData.X_train[:,1])
        x2_max_tr = np.amax(self.netData.X_train[:,1])  

        x1_min_te = np.amin(self.netData.X_test[:,0])
        x1_max_te = np.amax(self.netData.X_test[:,0])
        x2_min_te = np.amin(self.netData.X_test[:,1])
        x2_max_te = np.amax(self.netData.X_test[:,1]) 
        
        x1_min = 0
        x1_max = 0
        x2_min = 0
        x2_max = 0
        
        if x1_min_tr > x1_min_te:
            x1_min = x1_min_te
        else:
            x1_min = x1_min_tr

        print(x2_min,x2_min_tr,x2_min_te)
        if x1_max_tr > x1_max_te:
            x1_max = x1_max_tr
        else:
            x1_max = x1_max_te


        if x2_min_tr > x2_min_te:
            x2_min = x2_min_te
        else:
            x2_min = x2_min_tr
        print(x2_min,x2_min_tr,x2_min_te)


        if x2_max_tr > x2_max_te:
            x2_max = x1_max_tr
        else:
            x2_max = x2_max_te
            
        xx, yy = np.meshgrid(np.arange(x1_min,x1_max,0.01),np.arange(x2_min,x2_max,0.01))            

        z = self.predictModel(np.c_[xx.ravel(),yy.ravel()])
        z = z.reshape(xx.shape)
        
        plt.contour(xx,yy,z)
        plt.scatter(self.netData.X_train[:,0],self.netData.X_train[:,1],c=self.netData.y_train)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
        plt.show()

































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
