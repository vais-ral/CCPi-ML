# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:07:17 2018

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
flowers = ((np.asarray(Image.open('flowers.png').convert('L')))/255)*0.9

for a in range(0,1,1):
    pps = []
    back = []
    num_back_rectangles = np.random.randint(200, 400)
    for i in range(0,num_back_rectangles,1):
        pp = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
          'C0' : float(np.random.uniform(low=0.5, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        
        bb = {'Obj': TomoP2D.Objects2D.ELLIPSE, 
          'C0' : float(np.random.uniform(low=0.1, high=0.5, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        cc = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
          'C0' : float(np.random.uniform(low=0.5, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        dd = {'Obj': TomoP2D.Objects2D.ELLIPSE, 
          'C0' : float(np.random.uniform(low=0.5, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        back.append(pp)
        
        back.append(bb)
        
        back.append(cc)
       
        back.append(dd)
        
        back.append(bb)
    num_rectangles = np.random.randint(5, 20)
    print(num_rectangles)
    
    for i in range(0,num_rectangles,1):
        pp = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
          'C0' : float(np.random.uniform(low=0.7, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-0.5, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        
        bb = {'Obj': TomoP2D.Objects2D.ELLIPSE, 
          'C0' : float(np.random.uniform(low=0.7, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        cc = {'Obj': TomoP2D.Objects2D.RECTANGLE, 
          'C0' : float(np.random.uniform(low=0.7, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        dd = {'Obj': TomoP2D.Objects2D.ELLIPSE, 
          'C0' : float(np.random.uniform(low=0.7, high=0.9, size =1)), 
          'x0' : float(np.random.uniform(low=-1.0, high=1.0, size =1)),
          'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
          'a'  : float(np.random.uniform(low=0.02, high=0.3, size = 1)),
          'b'  : float(np.random.uniform(low=0.02, high=0.3, size=1)),
          'phi': float(np.random.uniform(low=0, high=180, size=1))}
        pps.append(pp)
        pps.append(bb)
        pps.append(cc)
        pps.append(dd)

    plt.close('all')
    
    shapes = np.zeros((256, 256), dtype=np.uint8)
    background = np.zeros((256, 256), dtype=np.uint8)
    
    for i in range(0,num_back_rectangles,1):
        phantom_2D = (TomoP2D.Object(256,back[i]))
        background = (background + phantom_2D)
    for i in range(0,num_rectangles,1):
        phantom_2D = (TomoP2D.Object(256,pps[i]))
        shapes = (shapes + phantom_2D)
          
    noise = np.random.uniform(low=0,high=1, size=(256, 256))
    filter1 = shapes >1
    shapes[filter1] = 1
    filter2 = background >1
    background[filter1] = 1
    IMAGE.append(shapes)
    nos = (background*1.0)+(noise*0.8)+(shapes*2) 
    scaler = np.amax(nos)
    nos = nos / scaler
    NOISY_IMAGE.append(nos)
    
    
#plt.figure(1)
#plt.rcParams.update({'font.size': 21}) 
NOISY_IMAGES = np.array(NOISY_IMAGE)
IMAGE = np.array(IMAGE)
#np.save('noisy_flower_experiment', NOISY_IMAGES)
#np.save('flowerexperiment.npy', IMAGE)
plt.imshow(NOISY_IMAGES[0],  cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('{}''{}'.format('2D Phantom using model no.',model))
#print(NOISY_IMAGE)
#print(IMAGE)