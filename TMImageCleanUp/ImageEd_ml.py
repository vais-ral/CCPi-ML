import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import keras
from skimage.filters.rank import median
from PIL import Image as tifIm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import math
from keras.models import load_model
import keras.backend as K

def openImage(path):
	return tifIm.open(path)

def saveImage(array,path):
    im = tifIm.fromarray(array.astype(np.uint16))
    im.save(path, "TIFF")

def image2Numpy(im):
    arr = np.array(im)
    return arr.astype(np.int32)

def scaleNumpy(arr,scale):
	return arr/scale

#Return Segment of image centered at x,y with width and height = segmentWidth
def imageSegment(image,x,y,segmentWidth):

	x_min = math.floor(x - (segmentWidth/2))
	x_max = math.floor(x + (segmentWidth/2))
	y_min = math.floor(y - (segmentWidth/2))
	y_max = math.floor(y + (segmentWidth/2))

	segment = image[int(y_min):int(y_max),int(x_min):int(x_max)]
    
	return segment

#Replace portion of image with segment centered at x,y where segment has width and height of segmentWidth
def replaceImageSegment(image,segment,x,y,segmentWidth):

    x_min =math.floor( x - (segmentWidth/2))
    x_max = math.floor(x + (segmentWidth/2))
    y_min = math.floor(y - (segmentWidth/2))
    y_max = math.floor(y + (segmentWidth/2))
  

    image[int(y_min):int(y_max),int(x_min):int(x_max)] = segment
    
    return image

#Remove hot pixel at x,y by applying median filter of size = width
def removeHotPixel(image,x,y,width):
    
    #Get segment of image at x,y
	segment = imageSegment(image,x,y,width)
    #Apply filter to segment
	filtered = ndimage.median_filter(segment,size=width)
    #Put segment back into image
	newImage = replaceImageSegment(image,filtered,x,y,width)
    
	return newImage

#Label Zingers given a threshold
def findZingers(image,threshold,width):
    
    #Apply Median Filter to entire image of size 10
    medImage = ndimage.median_filter(image.copy(),size=width)
    #Subtract filtered image from origianl image
    subImage = np.subtract(image,medImage)
    #Set all pixel's below 0 to 0
    filterZero = subImage <= 0
    subImage[filterZero] = 0
    #Search subtracted image for pixels above a certain value and make array of index's of these values
    index = np.argwhere(subImage > threshold)
    
    return index

#Clean an Image with no machine learning
def cleanImageNoML(image,maskSize,thresh):
    #Get Array of indexs where zingers are located
    index = findZingers(image.copy(),thresh,maskSize)
    #Copy image array
    newImage = image.copy()
    #Go through index loop and remove Hot pixel by applying small median filter to a specific area
    for i in index:
        temp = newImage.copy()
        newImage = removeHotPixel(newImage,i[1],i[0],maskSize)
        
    return newImage

#Build Model
def buildModel(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=input_shape,padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= keras.backend.image_data_format()))
#    
#    model.add(keras.layers.Conv2D(12, (3, 3),padding="same",data_format= keras.backend.image_data_format()))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= K.image_data_format()))
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
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    
    return model

#Train Model
def trainModel(model,features,labels,Epochs,BatchSize,LR):
    
    model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
    history = model.fit(features, labels, batch_size=BatchSize,epochs=Epochs, verbose=2)
  
    return history, model

#Generate array of padded image according to segment width and array of zinger indexes
def imageZingArray(image,segment,thresh,padding):
    #pad = 0 add padding, pad = 1 do not add padding
    if padding == 0:
        pad = imagePadArray(image,segment)
    else:
        pad = image
    return np.array([pad,findZingers(pad.copy(),thresh,segment)])

#Add padding to image
def imagePadArray(image,segment):
    return np.array(np.pad(image,segment,'constant', constant_values=0))


#Split Image into segments and assign it a label accroding to if it has a zinger
def buildSplitData(imZing,segment):
    segImages = []
    labels = []
    for x in range(int(segment),imZing[0].shape[0]-int(segment),segment):
        for y in range(int(segment),imZing[0].shape[1]-int(segment),segment):
            x_min = math.floor(x - (segment/2))
            x_max = math.floor(x + (segment/2))
            y_min = math.floor(y - (segment/2))
            y_max = math.floor(y + (segment/2))  
            #Look to see if current segment contains zinger in zinger list for each dimension
            XFilterMin = imZing[1][:,0] >= int(y_min)
            XFilterMax = imZing[1][:,0] <= int(y_max)
            YFilterMin = imZing[1][:,1] >= int(x_min)
            YFilterMax = imZing[1][:,1] <= int(x_max)         
            #Cross reffrence Xmin with Xmax see if a zinger is True for both
            XFilter = (np.all([XFilterMin,XFilterMax],axis=0))
            YFilter = (np.all([YFilterMin,YFilterMax],axis=0))
            #Cross reffrence X and Y is true to see is zinger was in both
            Filter = np.all([XFilter,YFilter],axis=0)
            #Labbel it if Filter is True
            if np.any(Filter):
                labels.append(1)
            else:
                labels.append(0)
            #Division by /65535 is for sclaiing pixel values
            segImages.append(imZing[0][int(y_min):int(y_max),int(x_min):int(x_max)]/65535)

    return np.array(segImages),np.array(labels)

def cleanImage(path,model,width,probThresh):

    img_rows, img_cols = width, width
    #Open image and convert to numpy array
    imageTif = openImage(path)
    imageArr = image2Numpy(imageTif)
    imageShape = imageArr.shape
    #Pad Image
    imageData = imagePadArray(imageArr,width) 
    #Split image
    splitImageData = buildSplitImage(imageData,width)  
    #copy original before data conversion for passing through network
    original = splitImageData.copy()
    #Convert image segments so that it is channel first or last
    if keras.backend.image_data_format() == 'channels_first':
        splitImageData = splitImageData.reshape(splitImageData.shape[0], 1, img_rows, img_cols)
    else:
        splitImageData = splitImageData.reshape(splitImageData.shape[0], img_rows, img_cols, 1)
    #Pass image through network get array of results for each image segment
    result = model.predict(splitImageData)
    #Look for results above a probability threshold
    filter1 = result >= probThresh  
    #Loop through filter list of index's and filter according segments  
    posImage = original.copy()
    index = (np.where(filter1))
    for seg in index[0]:
        posImage[seg] = np.ones(posImage[seg].shape)
        original[seg] = ndimage.median_filter(original[seg].copy(),size=width)    
    #Rebuild Image from segments
    image = rebuildImage(original,imageShape.shape[0],imageShape.shape[1],width)
    imagePos = rebuildImage(posImage,imageShape.shape[0],imageShape.shape[1],width)

    return image, imagePos

#Split image into array of segments with width and height = segment
def buildSplitImage(imZing,segment):
    
    segImages = []
    
    for x in range(int(segment),imZing.shape[0]-int(segment),segment):
        for y in range(int(segment),imZing.shape[1]-int(segment),segment):
            x_min = math.floor(x - (segment/2))
            x_max = math.floor(x + (segment/2))
            y_min = math.floor(y - (segment/2))
            y_max = math.floor(y + (segment/2))
            #Division by /65535 is for sclaiing pixel values
            segImages.append(imZing[int(y_min):int(y_max),int(x_min):int(x_max)]/65535)

    return np.array(segImages)

#Rebuild Image from segments
def rebuildImage(imageSeg,imX,imY,width):
    pad = width - (imX % width)
    
    imDimX = imX + width + pad
    imDimY = imY + width + pad
    shape = imageSeg.shape
    le = int(math.sqrt(shape[0]*shape[1]*shape[2]))
    image = np.zeros((le,le))
    counter = 0
    for arrX in np.arange(0,int(((math.sqrt(imageSeg.shape[0])))*width),width):
        for arrY in np.arange(0,int(((math.sqrt(imageSeg.shape[0])))*width),width):
            image[arrY:(arrY+width),arrX:(arrX+width)] = imageSeg[counter]
            counter+=1

    return image
#Build Labeled data set
def buildLabeledData(width,thresh):
    
    zingerDataCombinedIm = []
    zingerDataCombinedLabels = []
    
    #Loop through files
    for fileNum in range(0,21):
        #File name
        file = "%03d" % (fileNum)
        path = 'Images/Chamber_Flange_Tomo_'
        print(fileNum,file)
        #Open image and convert to numpy array
        imageTif = openImage(path+file+".tif")
        imageArr = image2Numpy(imageTif)
        #Generate padded image and list of zingers
        zingerData = imageZingArray(imageArr,width,thresh,0)
        #Generate split image and labels associated with each segments
        zingerDataIm, zingerDataLabel = buildSplitData(zingerData,width)
        zingerDataCombinedIm.append(zingerDataIm)
        zingerDataCombinedLabels.append(zingerDataLabel)
        
    pathF = 'Data/Chamber_Flange_Tomo_Features'
    pathL = 'Data/Chamber_Flange_Tomo_Labels'
    #Save data
    np.save(pathF,np.array(zingerDataCombinedIm).reshape(len(zingerDataCombinedIm)* zingerDataCombinedIm[0].shape[0], width, width))
    np.save(pathL,np.array(zingerDataCombinedLabels).reshape(len(zingerDataCombinedIm)* zingerDataCombinedIm[0].shape[0]))

#Load training data
def loadTrainingData(fileF,fileL):
    return np.load(fileF),np.load(fileL)
#Shuffle Data
def shuffleData(features,labels):
    
    ran = np.arange(features.shape[0])
    np.random.shuffle(ran)
    features= features[ran]
    labels= labels[ran]
    
    return features,labels

#Split data
def splitData(features,labels,ratio):
    length = features.shape[0]
    return features[:int(length*ratio)],features[:int(length*(1-ratio))],labels[:int(length*ratio)],labels[:int(length*(1-ratio))]

#Load Model
<<<<<<< HEAD
def loadModel(name):
    model = load_model(name)
=======
def loadModel():
    model = load_model('TMML_Model_10_10_conv64_pool_dense200.h5')
>>>>>>> parent of 7cf8ee2... Update
    return model

#Save Model
def saveModel(model,name):

    save = "d"
    
    while(save != "y" or save != "n"):

        save = input("Do you want to save the model (y/n)?")
    
        if save == "y":
            model.save(name)
            break
        elif save == "n":
            break

#Generate new model or save model
def loadOrGenModel(input_shape,name):
    
    load = "d"
    
    while(load != "y" or load != "n"):

        load = input("Load existing model (y/n)?")
    
        if load == "y":
            return loadModel(name)
        elif load == "n":  
            return buildModel(input_shape,1)

#Rebalance data set
def reBalanceData(x,y,Multip):
    ones = x[np.where(y==1)].copy()
    y_ones = y[np.where(y==1)].copy()
    total = len(y)
    total_one = len(ones)
    multiplier = int(math.ceil((total/total_one)*Multip))
    for i in range(multiplier):
        x = np.insert(x,1,ones,axis=0)
        y = np.insert(y,1,y_ones,axis=0)

    ran = np.arange(x.shape[0])
    np.random.shuffle(ran)
    x= x[ran]
    y= y[ran]
    return x,y  

#Data formating NCWH or NWHC
def channelOrderingFormat(Feat_train,Feat_test,img_rows,img_cols):
    if keras.backend.image_data_format() == 'channels_first':
        Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
        Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
        Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)  
    return Feat_train, Feat_test, input_shape
##########################################################################################





############################################################################################
    
#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.tensorflow_backend._get_available_gpus()
#%%

#################Settings###########################

imageTypeMulti = 65535

trainFeatPath = r'C:\Users\lhe39759\Documents\GitHub\Data_TM\Chamber_Flange_Tomo_Features_10_10.npy'
trainLabelPath = r'C:\Users\lhe39759\Documents\GitHub\Data_TM\Chamber_Flange_Tomo_Labels_10_10.npy'

loadModelName = 'TMML_Model.h5'
saveModelName = 'TMML_Model.h5'

rebalanceRatio = 1
splitDataRatio = 0.6

splitDim = 10

Epochs = 10
miniBatch = 10000
Lr = 0.00003

######################################################
#%%
#buildLabeledData(30,3*10**3)
#%%
    
Features,Labels = loadTrainingData(trainFeatPath,trainLabelPath)

Features = Features.reshape(21* 42025, 10, 10)
Labels = Labels.reshape(21*42025)
Features,Labels = shuffleData(Features,Labels)

Features,Labels = reBalanceData(Features,Labels,rebalanceRatio)


<<<<<<< HEAD
Feat_train,Feat_test,Labels_train,Labels_test = splitData(Features,Labels,splitDataRatio) 
=======
Feat_train,Feat_test,Labels_train,Labels_test = splitData(Features,Labels,0.1) 
>>>>>>> parent of 7cf8ee2... Update
#%%

#Data formating NCWH or NWHC
img_rows, img_cols = splitDim, splitDim
Feat_train,Feat_test,input_shape = channelOrderingFormat(Feat_train,Feat_test,img_rows,img_cols)

#%%

model = loadOrGenModel(input_shape,loadModelName)
print(Feat_train.shape)
history, model = trainModel(model,Feat_train,Labels_train,Epochs,miniBatch,Lr)
#scores = model.evaluate(Feat_test,Labels_test)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
saveModel(model,saveModelName)

#%%

image, posImage = cleanImage('D:\Documents\Dataset\Chadwick_Flange_Tomo\Data\Chamber_Flange_Tomo_2202.tif',model,10,1)
saveImage(image*imageTypeMulti,'test.tif')
saveImage(posImage*imageTypeMulti,'test_pos.tif')

