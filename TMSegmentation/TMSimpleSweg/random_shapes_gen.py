# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:07:17 2018

@author: zyv57124
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL
#import PIL
#from PIL import Image
from tomophantom import TomoP2D

def genTomoShape(typeShape):
    return {'Obj': typeShape, 
              'C0' : float(np.random.uniform(low=1.0, high=1.0, size =1)), 
              'x0' : float(np.random.uniform(low=-0.5, high=1.0, size =1)),
              'y0' : float(np.random.uniform(low=-1.0, high=0.5, size =1)),
              'a'  : float(np.random.uniform(low=0.1, high=0.4, size = 1)),
              'b'  : float(np.random.uniform(low=0.1, high=0.4, size=1)),
              'phi': float(np.random.uniform(low=0, high=180, size=1))}

def generateImage():
    model = 6 # selecting a model
    N_size = 256
    #specify a full path to the parameters file
    pathTP = 'Phantom2DLibrary.dat'
    #This will generate a N_size x N_size phantom (2D)
    phantom_2D = TomoP2D.Model(model, N_size, pathTP)
    
    Labels = []
    Labels_array = []
    Images = []
    shapeTypes = [TomoP2D.Objects2D.RECTANGLE,TomoP2D.Objects2D.ELLIPSE]
    
    
    insertImage = np.asarray(PIL.Image.open('2.jpg').convert('L'))
#    print(inssertImage.shape)
#    insertImage = np.append(insertImage,insertImage, axis=0)
#    insertImage = np.append(insertImage,insertImage, axis=0)
#    insertImage = np.append(insertImage,insertImage, axis=0)
#    print(insertImage.shape)
    insertImage.setflags(write=1)

    insertImage=[insertImage[:256,:256],insertImage[:256,:256]]
    
    for a in range(0,1,1):
        
        objects = []
        background = []
        array_label = []
        num_back_shapes = np.random.randint(200, 400)
        
        for i in range(0,num_back_shapes,1):
            shapeRand = np.random.randint(2) 
            background.append(genTomoShape(shapeTypes[shapeRand]))
    
        num_shapes = np.random.randint(5, 20)
        num_shapes = 5
        
        label_temp = np.zeros((256, 256), dtype=np.uint8)
        image_temp = np.zeros((256, 256), dtype=np.uint8)
        
        pattern = np.random.randint(len(insertImage))
        
        for i in range(0,num_shapes,1):
    
            #Generate tomoshape
            #Random pick of shape type
            shapeRand = np.random.randint(2) 
            #Generate shape            
            shape = TomoP2D.Object(256, genTomoShape(shapeTypes[shapeRand]))
            #add the shape to the laebl image and take away the original to get the positions where a new shape has been added. 
            fit = ((label_temp+shape) - image_temp) > 0
            #update the pixel values where the shape has been added, do no tjust add the pixel values together but replace them so the overlap is smooth
            label_temp[fit] = shape[fit]
            #filter looking for where the shape is
            shapeFilter = shape >= 0.5
            
            #for those positions where the shape exists set those values to the value in the image pattern
            shape[shapeFilter] = insertImage[pattern][shapeFilter]
            #add the shape to the image and take away the original to get the positions where a new shape has been added. 
            fit = ((image_temp+shape) - image_temp) > 0
            #update the pixel values where the shape has been added, do no tjust add the pixel values together but replace them so the overlap is smooth
            image_temp[fit] = shape[fit]
        
        array_temp = np.zeros(len(insertImage))
        array_temp[pattern] = int(1)
        array_label.append(array_temp)
     
        noise = np.random.uniform(low=0,high=np.amax(image_temp)*0.1, size=(256, 256))
        nos = (image_temp*1.0)+(noise*1.0) 
        plt.imshow(nos)
        Images.append(nos)
        
        #%%
        
#    fig = plt.figure(figsize=(15, 15))
#    fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
#    #plt.figure(1)
#    #plt.rcParams.update({'font.size': 21}) 
##    NOISY_IMAGES = np.array(NOISY_IMAGE)
##    IMAGE = np.array(IMAGE)
##    np.save('TM_Images.npy', NOISY_IMAGES)
##    np.save('TM_Labels.npy', IMAGE)
#    ax_heat = fig.add_subplot(121)
#
#    ax_heat.imshow(NOISY_IMAGE[0],  cmap="BuPu")
#    ax_heat = fig.add_subplot(122)
#
#    ax_heat.imshow(IMAGE[0],  cmap="BuPu")
#
#    plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
#    plt.title('{}''{}'.format('2D Phantom using model no.',model))
#    plt.show()
    return np.array([Labels,Images])

generateImage()
plt.show()

#    #print(NOISY_IMAGE)
#    #print(IMAGE)