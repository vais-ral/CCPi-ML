# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:26:22 2018

@author: lhe39759
"""
import math
import numpy as np


def reBalanceData(x,y,Multip):
       
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
    

def splitData(features,labels,ratio):
    length = features.shape[0]
    return features[:int(length*ratio)],features[:int(length*(1-ratio))],labels[:int(length*ratio)],labels[:int(length*(1-ratio))]

def shuffleData(features,labels):
    
    ran = np.arange(features.shape[0])
    np.random.shuffle(ran)
    features= features[ran]
    labels= labels[ran]
    
    return features,labels

def imagePadArray(image,segment):
    return np.array(np.pad(image,segment,'constant', constant_values=0))