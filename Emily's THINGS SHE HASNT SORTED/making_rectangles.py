# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 09:33:18 2018

@author: zyv57124
"""

## 1 rectangular
#Model : 06;
#Components : 01;
#TimeSteps : 1;
#Object : rectangle 1.00 -0.15 0.2 0.4 0.3 45;

import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import PIL
from PIL import Image

model = 6 # selecting a model
N_size = 100
#specify a full path to the parameters file
pathTP = r'C:\Users\zyv57124\Documents\TomoPhantom-master\TomoPhantom-master\functions\models\Phantom2DLibrary.dat'
#This will generate a N_size x N_size phantom (2D)
phantom_2D = TomoP2D.Model(model, N_size, pathTP)

#pp = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
#      'C0' : 1.0, 
#      'x0' : 0.25,
#      'y0' : 0.37,
#      'a'  : 1.3,
#      'b'  : 0.02,
#      'phi': 90.0}

#create lists of 10 random values for each
#for x0 in range(-0.25,0.25, 0.01)
#for y0 in range(-0.37, 0.37)
#for a in range(0.1, 1.3, 0.02)
#for b in range(0.02, 0.05, 0.002)
#for phi in range(0,180, 5)

NOISY_IMAGE = []
NOISE_ALONE = []

for a in range(0,1,1):
    pps = []
    for i in range(0,50,1):
        pp = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
          'C0' : 1.0, 
          'x0' : float(np.random.uniform(low=-0.25, high=0.25, size =1)),
          'y0' : float(np.random.uniform(low=-0.37, high=0.37, size =1)),
          'a'  : float(np.random.uniform(low=0.1, high=1.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.05, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        pps.append(pp)
    
    plt.close('all')
    
    #plt.figure(1)
    #plt.rcParams.update({'font.size': 21})
     
    array = np.zeros((100, 100), dtype=numpy.uint8)
    
    for i in range(0,50,1):
        phantom_2D = (TomoP2D.Object(100,pps[i]))
        array = (array + phantom_2D)
       
    noise = numpy.random.uniform(low=0,high=0.8, size=(100, 100))
    
    #NOISY_IMAGE.append()
    
    plt.imshow(array+noise, vmin=0, vmax=1.5, cmap="BuPu")
    #plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
    #plt.title('{}''{}'.format('2D Phantom using model no.',model))
