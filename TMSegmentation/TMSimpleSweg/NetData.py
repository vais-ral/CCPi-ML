# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:26:22 2018

@author: lhe39759
"""
import math
import numpy as np
import keras
class NetData:

    def __init__(self,features,labels,Shuffle=True,Rebalance = 0.0, Split_Ratio = 0.7):

        if Shuffle:
            features ,labels = self.shuffleData(features,labels)

        self.X_train,self.X_test,self.y_train,self.y_test = self.splitData(features,labels,Split_Ratio)

        if Rebalance != None:
            self.X_train,self.y_train = self.reBalanceData(self.X_train,self.y_train,Rebalance)

    def channelOrderingFormatTrain(self,img_rows,img_cols):
        self.X_train,self.y_train,input_shape = self.channelOrderingFormat(self.X_train,self.y_train,X_train,self.y_train,img_rows,img_cols)
        return input_shape
        
    def channelOrderingFormatTest(self,img_rows,img_cols):
        self.X_test, self.y_test,input_shape = self.channelOrderingFormat(self.X_test, self.y_test,img_rows,img_cols)
        return input_shape

    def channelOrderingFormat(self,Feat_train,Feat_test,img_rows,img_cols):
        if keras.backend.image_data_format() == 'channels_first':
            Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)  
        return Feat_train, Feat_test, input_shape

    def reBalanceData(self,x,y,Multip):
        
            ones = x[np.where(y==1)].copy()
            y_ones = y[np.where(y==1)].copy()
            total = len(y)
            total_one = len(ones)
            multiplier = int(math.ceil((total/total_one)*Multip))
            print(multiplier,ones.shape,x.shape)
            for i in range(multiplier):
                x = np.insert(x,1,ones,axis=0)
                y = np.insert(y,1,y_ones,axis=0)
        
            ran = np.arange(x.shape[0])
            np.random.shuffle(ran)
            
            return x[ran], y[ran]
        

    def splitData(self,features,labels,ratio):
        length = features.shape[0]
        return features[:int(length*ratio)],features[:int(length*(1-ratio))],labels[:int(length*ratio)],labels[:int(length*(1-ratio))]

    def shuffleData(self,features,labels):
        
        ran = np.arange(features.shape[0])
        np.random.shuffle(ran)
        features= features[ran]
        labels= labels[ran]
        
        return features,labels

    def imagePadArray(self,image,segment):
        return np.array(np.pad(image,segment,'constant', constant_values=0))