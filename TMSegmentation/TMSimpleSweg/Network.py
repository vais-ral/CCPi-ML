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
#    {"Input": tuple for convolution input, int for data,
#    "HiddenLayers": List of hiddenlayers see below for structure, avalible options are:
#        Convolution ->
#        
#         {"Type": 0,
#            "Width": Layer Width, int E.G 10
#            "Activation": Activation type E.G "relu","tanh","sigmoid",
#            "Kernel": Kernal Size, tuple E.G (3,3)
#            }
#        Dense ->
#        
#         {"Type": 1,
#            "Width": Layer Width, int E.G 10,
#            "Activation":  Activation type E.G "relu","tanh","sigmoid""
#            }
#         
#        Dropout ->
#            {"Type": 2,
#            "Ratio": Ratio of neurons to drop out, float E.G 0.7
#            }
#        
#        Pooling -> 
#        {"Type": 3,
#        "kernal": Kernal Size, tuple E.G (3,3)}
#        } 
#   
#       Flattern - >
#        {"Type": 4,
#        } 
#    }    
#
#    """
    #CANT HAVE FLATTERN AS FIRST LAYER YET    
    
    def __init__(self,Network,Backend):
        self.input = Network["Input"]
        self.hiddenLayers = Network["HiddenLayers"]
        self.backend = Backend
        self.model = self.buildModel()
        
    def buildModel(self):
        if self.backend == 'keras':
            return self.kerasModelBackend()
 
    def loadModel(self,name):
        if self.backend == 'keras':
            return keras.models.load_model(name)
        
    def saveModel(self,name):
        if self.backend == 'keras':
            self.model.save(name)

    def compileModel(self,LR):
        if self.backend== 'keras':
            self.kerasCompileModel(LR)
                
    def trainModel(self,features,labels,Epochs,BatchSize,LR):
        if self.backend== 'keras':
            self.compileModel(LR)
            return self.kerasTrainModel(features,labels,Epochs,BatchSize)    
        
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
            for layer in range(0,len(self.hiddenLayers)):    
                print('type',self.hiddenLayers[layer]["Type"])
                #Check for first layer to deal with input shape
                
                if layer == 0:
                    #Convolution first layer
                    if self.hiddenLayers[layer]["Type"] == 0:
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], input_shape=(self.input),padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                    #Dense first layer
                    elif self.hiddenLayers[layer]["Type"] == 1:
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],input_dim=(self.input),kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))
                else:
                # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flattern 
                    if self.hiddenLayers[layer]["Type"] == 0:
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                    elif self.hiddenLayers[layer]["Type"] == 1:
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))       
                    elif self.hiddenLayers[layer]["Type"] == 2:
                        layers.append(keras.layers.Dropout(self.hiddenLayers[layer]["Ratio"]))    
                    elif self.hiddenLayers[layer]["Type"] == 3:
                        layers.append(keras.layers.MaxPooling2D(pool_size=self.hiddenLayers[layer]["Kernal"],data_format= keras.backend.image_data_format()))
                    elif self.hiddenLayers[layer]["Type"] == 4:
                        layers.append(keras.layers.Flatten())
            return keras.Sequential(layers)
            

    def kerasCompileModel(self,LR):
        self.model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
        
    def kerasTrainModel(self,features,labels,Epochs,BatchSize):    
        return self.model.fit(features, labels, batch_size=BatchSize,epochs=Epochs, verbose=2)
      
            
    def kerasChannelOrderingFormat(self,Feat_train,Feat_test,img_rows,img_cols):
        if keras.backend.image_data_format() == 'channels_first':
            Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)  
        return Feat_train, Feat_test, input_shape
    
#################################################################


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
        


