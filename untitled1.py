# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:16:32 2018

@author: zyv57124
"""

import numpy as np

import matplotlib.pyplot as plt

from tomophantom import TomoP2D



model = 1 # selecting a model

N_size = 512

#specify a full path to the parameters file

pathTP = r'C:\Users\zyv57124\Documents\TomoPhantom-master\TomoPhantom-master\functions\models\Phantom2DLibrary.dat'

#objlist = modelfile2Dtolist(pathTP, model) # one can extract parameters

#This will generate a N_size x N_size phantom (2D)

phantom_2D = TomoP2D.Model(model, N_size, pathTP)



plt.close('all')

plt.figure(1)

plt.rcParams.update({'font.size': 21})

plt.imshow(phantom_2D, vmin=0, vmax=1, cmap="BuPu")

plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')

plt.title('{}''{}'.format('2D Phantom using model no.',model))
