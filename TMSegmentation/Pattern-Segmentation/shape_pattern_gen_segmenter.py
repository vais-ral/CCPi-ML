# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:07:17 2018

@author: zyv57124
"""
import os
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
              'a'  : float(np.random.uniform(low=0.2, high=0.4, size = 1)),
              'b'  : float(np.random.uniform(low=0.2, high=0.4, size=1)),
              'phi': float(np.random.uniform(low=0, high=180, size=1))}
def loadImg():
    
    path = "Patterns/"
    patOptions = ["annealing_twins","Brass bronze","Ductile_Cast_Iron","Grey_Cast_Iron","hypoeutectoid_steel","malleable_cast_iron","superalloy"]
    
    image_array = []
    
    for folder in patOptions:
        folder_array = []
        for filename in os.listdir(path+folder+"/"):
            if filename.endswith(".png"):
                insertImage1 = np.asarray(PIL.Image.open(path+folder+"/"+filename).convert('L'))
                insertImage1.setflags(write=1)
                insertImage1 = np.pad(insertImage1, (300,300), 'symmetric')
                folder_array.append(np.array(insertImage1[:256,:256]))
        image_array.append(np.array(folder_array))

    return (np.array(image_array))
    
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
    
    
    insertImage=loadImg()
    
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
        
        insertImage = loadImg() 
        
        label_temp = np.zeros((256, 256), dtype=np.uint8)
        image_temp = np.zeros((256, 256), dtype=np.uint8)
        
        temp_label_array = []
        
        for i in range(0,len(insertImage)):
            temp_label_array.append(label_temp)
            
        temp_label_array = np.array(temp_label_array)  


        for i in range(0,num_shapes,1):
            patternType = np.random.randint((insertImage.shape[0]))
            pattern = np.random.randint((insertImage[patternType].shape[0]))

            #Random pick of shape type
            shapeRand = np.random.randint(2) 
            #Generate shape            
            shape = TomoP2D.Object(256, genTomoShape(shapeTypes[shapeRand]))
            #add the shape to the laebl image and take away the original to get the positions where a new shape has been added. 
            fit = ((temp_label_array[patternType]+shape) - temp_label_array[patternType]) > 0
            #update the pixel values where the shape has been added, do no tjust add the pixel values together but replace them so the overlap is smooth
            temp_label_array[patternType][fit] = shape[fit]
            #filter looking for where the shape is
            shapeFilter = shape >= 0.5
            

            #for those positions where the shape exists set those values to the value in the image pattern
            shape[shapeFilter] = insertImage[patternType][pattern][shapeFilter]
            #add the shape to the image and take away the original to get the positions where a new shape has been added. 
            fit = ((image_temp+shape) - image_temp) > 0
            #update the pixel values where the shape has been added, do no tjust add the pixel values together but replace them so the overlap is smooth
            image_temp[fit] = shape[fit]

     
        noise = np.random.uniform(low=0,high=np.amax(image_temp)*0.1, size=(256, 256))
        nos = (image_temp*1.0)+(noise*1.0)

        Images.append(nos)
        Labels_array.append(np.array(temp_label_array))
        
        #%%
        item = []
        item.append(Images)
        item.append(Labels_array)
        item = np.array(item)
        
    return np.array(nos),np.array(temp_label_array)
f,l =  generateImage()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(f)
ax = fig.add_subplot(122)
print(l.shape)
ax.imshow(l[0])
plt.show()