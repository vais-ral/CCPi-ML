import tensorflow as tf
import keras
from skimage.filters.rank import median
from PIL import Image as tifIm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import math
from keras import backend as K

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

def imageSegment(image,x,y,segmentWidth):

	x_min = math.floor(x - (segmentWidth/2))
	x_max = math.floor(x + (segmentWidth/2))
	y_min = math.floor(y - (segmentWidth/2))
	y_max = math.floor(y + (segmentWidth/2))

	segment = image[int(y_min):int(y_max),int(x_min):int(x_max)]
    
	return segment

def replaceImageSegment(image,segment,x,y,segmentWidth):

    x_min =math.floor( x - (segmentWidth/2))
    x_max = math.floor(x + (segmentWidth/2))
    y_min = math.floor(y - (segmentWidth/2))
    y_max = math.floor(y + (segmentWidth/2))
  

    image[int(y_min):int(y_max),int(x_min):int(x_max)] = segment
    
    return image

def removeHotPixel(image,x,y,width):

	segment = imageSegment(imageArr,x,y,width)
	filtered = ndimage.median_filter(segment,size=width)
	newImage = replaceImageSegment(imageArr,filtered,x,y,width)
    
	return newImage

def findZingers(image,threshold):
    
    medImage = ndimage.median_filter(image.copy(),size=10)
    subImage = np.subtract(image,medImage)
    filterZero = subImage <= 0
    subImage[filterZero] = 0
    plt.imshow(subImage)
    plt.show()
    index = np.argwhere(subImage > threshold)
    
    return index

def cleanImage(image,maskSize,thresh):
    
    index = findZingers(image.copy(),thresh)
    newImage = image.copy()

    for i in index:
        temp = newImage.copy()
        newImage = removeHotPixel(newImage,i[1],i[0],maskSize)
        
    return newImage

def buildModel(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(12, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100,activation='tanh'))
    #model.add(keras.layers.Dense(14,activation='tanh'))

#    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
    
    return model

def trainModel(model,features,labels,Epochs,BatchSize,LR):
    
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])
    history = model.fit(features, labels, batch_size=BatchSize,epochs=Epochs, verbose=2)
  
    return history, model

def imageZingArray(image,segment,thresh,padding):
    #pad = 0 add padding 1 do not
    if padding == 0:
        pad = np.pad(image,segment,'constant', constant_values=0)
    else:
        pad = image
    return np.array([pad,findZingers(pad.copy(),thresh)])


def buildSplitData(imZing,segment):
    segImages = []
    labels = []
    for x in range(int(segment),imZing[0].shape[0]-int(segment),segment):
        for y in range(int(segment),imZing[0].shape[1]-int(segment),segment):
            x_min = math.floor(x - (segment/2))
            x_max = math.floor(x + (segment/2))
            y_min = math.floor(y - (segment/2))
            y_max = math.floor(y + (segment/2))
            XFilterMin = imZing[1][:,0] >= int(y_min)
            XFilterMax = imZing[1][:,0] <= int(y_max)
            YFilterMin = imZing[1][:,1] >= int(x_min)
            YFilterMax = imZing[1][:,1] <= int(x_max)
            
            XFilter = (np.all([XFilterMin,XFilterMax],axis=0))
            YFilter = (np.all([YFilterMin,YFilterMax],axis=0))
            Filter = np.all([XFilter,YFilter],axis=0)
            if np.any(Filter):
                labels.append(1)
            else:
                labels.append(0)
            #Division by /65535 is for sclaiing pixel values
            segImages.append(imZing[0][int(y_min):int(y_max),int(x_min):int(x_max)]/65535)
    print(np.array(labels).shape,np.sum(np.array(labels) == 0))

    return np.array(segImages),np.array(labels)


def buildSplitImage(imZing,segment):
    
    segImages = []
    
    for x in range(int(segment),imZing[0].shape[0]-int(segment),segment):
        for y in range(int(segment),imZing[0].shape[1]-int(segment),segment):
            x_min = math.floor(x - (segment/2))
            x_max = math.floor(x + (segment/2))
            y_min = math.floor(y - (segment/2))
            y_max = math.floor(y + (segment/2))
            #Division by /65535 is for sclaiing pixel values
            segImages.append(imZing[0][int(y_min):int(y_max),int(x_min):int(x_max)]/65535)

    return np.array(segImages)

def rebuildImage(imageSeg,imX,imY,width):
    pad = width - (imX % width)
    
    imDimX = imX + width + pad
    imDimY = imY + width + pad
    shape = imageSeg.shape
    le = int(math.sqrt(shape[0]*shape[1]*shape[2]))
    image = np.zeros((le,le))

    counter = 0
    print(imageSeg.shape,width)
    for arrX in np.arange(0,int(((math.sqrt(imageSeg.shape[0])))*width),width):
        for arrY in np.arange(0,int(((math.sqrt(imageSeg.shape[0])))*width),width):
            print(arrX,arrY)
            image[arrY:(arrY+width),arrX:(arrX+width)] = imageSeg[counter]
            counter+=1

    return image

def buildLabeledData(width,thresh):
    
    zingerDataCombinedIm = []
    zingerDataCombinedLabels = []
    
    for fileNum in range(0,21):
        
        file = "%03d" % (fileNum)
        path = 'Images/Chamber_Flange_Tomo_'
        print(fileNum,file)
        imageTif = openImage(path+file+".tif")
        imageArr = image2Numpy(imageTif)
        plt.imshow(imageArr)
        plt.show()
        zingerData = imageZingArray(imageArr,width,thresh,0)
        zingerDataIm, zingerDataLabel = buildSplitData(zingerData,width)
        zingerDataCombinedIm.append(zingerDataIm)
        zingerDataCombinedLabels.append(zingerDataLabel)
        
    pathF = 'Data/Chamber_Flange_Tomo_Features'
    pathL = 'Data/Chamber_Flange_Tomo_Labels'
  
    np.save(pathF,np.array(zingerDataCombinedIm))

def loadTrainingData(fileF,fileL):
    return np.load(fileF),np.load(fileL)

def shuffleData(features,labels):
    
    ran = np.arange(features.shape[0])
    np.random.shuffle(ran)
    features= features[ran]
    labels= labels[ran]
    
    return features,labels
    
def splitData(features,labels,ratio):
    
    length = features.shape[0]
    return features[:int(length*ratio)],features[:int(length*(1-ratio))],labels[:int(length*ratio)],labels[:int(length*(1-ratio))]
   
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
#buildLabeledData(10,3*10**3)

#%%
Features,Labels = loadTrainingData('Data/Chamber_Flange_Tomo_Features.npy','Data/Chamber_Flange_Tomo_Labels.npy')

Features = Features.reshape(21* 42025, 10, 10)
Labels = Labels.reshape(21*42025)
Features,Labels = shuffleData(Features,Labels)

Features,Labels = reBalanceData(Features,Labels,1.0)


img_rows, img_cols = 10, 10
print(Labels.shape,np.sum(Labels == 0))

Feat_train,Feat_test,Labels_train,Labels_test = splitData(Features,Labels,0.6) 
#%%
if K.image_data_format() == 'channels_first':
    Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
    Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
    Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print(Features[1].shape)
plt.imshow(Features[1])

plt.show()

model = buildModel(input_shape,1)

history, model = trainModel(model,Feat_train,Labels_train,10,10000,0.01)
scores = model.evaluate(Feat_test,Labels_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#%%
file = "%03d" % (1)
path = 'Images/Chamber_Flange_Tomo_'
imageTif = openImage(path+file+".tif")
imageArr = image2Numpy(imageTif)
zingerData = imageZingArray(imageArr,10,3.3*10**3,0)
plt.imshow(zingerData[0]/65535)
plt.show()
zingerDataIm, zingerDataLabel = buildSplitData(zingerData,10)

#%%
rebuild = rebuildImage(zingerDataIm,2048,2048,10)
print(rebuild.shape)
plt.imshow(rebuild)
plt.show()