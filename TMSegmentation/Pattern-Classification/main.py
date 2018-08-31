# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:33:05 2018

@author: lhe39759
"""
import keras
import os
import PIL
import sys
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub/')
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


def generateData():
    images = loadImg()
    features = []
    labels = []
    
    for folder in range(0,images.shape[0]):
        for image in images[folder]:
            features.append(image)
            labels.append(folder)
            
    return np.array(features),np.array(labels).reshape(len(labels),1)
            
            
features, labels = generateData()
#%%
model = keras.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(256,256,1),padding="same",data_format= keras.backend.image_data_format()))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= keras.backend.image_data_format()))
#    
model.add(keras.layers.Conv2D(32, (3, 3),padding="same",data_format= keras.backend.image_data_format()))

##    
#    model.add(keras.layers.Conv2D(64, (2, 2),data_format= K.image_data_format()))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= K.image_data_format()))
model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(keras.layers.Dense(200))
model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.Dense(64))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(7))
model.add(keras.layers.Activation('sigmoid'))
model.add(keras.layers.Softmax())
print(features.shape,labels.shape)
data = DataSlice(Features = features, Labels = labels,Shuffle=True,Split_Ratio = 0.7,Channel_Features= (256,256))
data.oneHot(7)
model = NetSlice(model,'keras', data)
#model.loadModel('modelSaveTest',customObject={'bce_dice_loss':bce_dice_loss,'dice_coeff':dice_coeff})
model.compileModel(keras.optimizers.RMSprop(lr=0.01), 'categorical_crossentropy', [dice_coeff])
model.trainModel(Epochs = 2,Batch_size = 1, Verbose = 2)