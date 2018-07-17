# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:36:49 2018

@author: lhe39759
"""

from mxnet import nd
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
import numpy as np
def Numpy2MXNet(arr):
	return np.array(arr)

a = np.array([2,3,4])
print(len(np.where(a==2)))
layer = nn.Dense(2)
layer.initialize()

net = nn.Sequential()

with net.name_scope():
	net.add(
	
	nn.Dense(400,activation="relu"),
	nn.Dense(25,activation="relu"),
	nn.Dense(10)
	
	)