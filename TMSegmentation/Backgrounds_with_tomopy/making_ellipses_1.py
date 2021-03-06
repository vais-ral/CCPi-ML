# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:44:00 2018

@author: zyv57124
"""

import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import PIL
from PIL import Image

model = 6 # selecting a model
N_size = 256
#specify a full path to the parameters file
pathTP = r'C:\Users\zyv57124\Documents\TomoPhantom-master\TomoPhantom-master\functions\models\Phantom2DLibrary.dat'
#This will generate a N_size x N_size phantom (2D)
phantom_2D = TomoP2D.Model(model, N_size, pathTP)

IMAGE = []
NOISY_IMAGE = []

for a in range(0,10000,1):
    pps = []
    num_rectangles = np.random.randint(1, 11)
    for i in range(0,num_rectangles,1):
        pp = {'Obj': TomoP2D.Objects2D.GAUSSIAN, 
          'C0' : 1.0, 
          'x0' : float(np.random.uniform(low=-0.5, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        pps.append(pp)
    
    plt.close('all')
    
    array = np.zeros((256, 256), dtype=np.uint8)
    
    for i in range(0,num_rectangles,1):
        phantom_2D = (TomoP2D.Object(256,pps[i]))
        array = (array + phantom_2D)
       
    noise = np.random.uniform(low=0,high=1.5, size=(256, 256))
    filter1 = array >1
    array[filter1] = 1
    IMAGE.append(array)
    nos = array+noise
    scaler = np.amax(nos)
    nos = nos / scaler
    NOISY_IMAGE.append(nos)
    
    
#plt.figure(1)
#plt.rcParams.update({'font.size': 21}) 
NOISY_IMAGE = np.array(NOISY_IMAGE)
IMAGE = np.array(IMAGE)
np.save('nosey_ellipses.npy', NOISY_IMAGE)
np.save('ellipses.npy', IMAGE)
#plt.imshow(array+noise, vmin=0, vmax=1, cmap="BuPu")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
#plt.title('{}''{}'.format('2D Phantom using model no.',model))
#print(NOISY_IMAGE)
#print(IMAGE)