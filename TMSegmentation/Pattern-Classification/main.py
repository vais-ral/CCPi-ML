# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:33:05 2018

@author: lhe39759
"""

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