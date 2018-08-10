# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:30:25 2018

@author: zyv57124
"""

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

IMAGE = []
NOISY_IMAGE = []

for a in range(0,100000,1):
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
    
    array = np.zeros((100, 100), dtype=numpy.uint8)
    
    for i in range(0,50,1):
        phantom_2D = (TomoP2D.Object(100,pps[i]))
        array = (array + phantom_2D)
       
    noise = numpy.random.uniform(low=0,high=0.8, size=(100, 100))
    
    IMAGE.append(array)
    NOISY_IMAGE.append(array+noise)
    
NOISY_IMAGE = np.array(NOISY_IMAGE)
IMAGE = np.array(IMAGE)
np.save('nosey_images.npy', NOISY_IMAGE)
np.save('images.npy', IMAGE)
    #plt.imshow(array+noise, vmin=0, vmax=1.5, cmap="BuPu")
   
