


from skimage.filters.rank import median
from PIL import Image as tifIm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc

def openImage(path):
	return tifIm.open(path)

def saveImage(array,path):
	im = tifIm.fromarray(array) # float32
	im.save(path, "TIFF")

def image2Numpy(im):
	return np.array(im)

def scaleNumpy(arr,scale):
	return arr/scale

def imageSegment(image,x,y,segmentWidth):

	x_min = x - (segmentWidth/2)
	x_max = x + (segmentWidth/2)
	y_min = y - (segmentWidth/2)
	y_max = y + (segmentWidth/2)
	segment = image[int(y_min):int(y_max),int(x_min):int(x_max)]
	return segment

def replaceImageSegment(image,segment,x,y,segmentWidth):
	print(x,y,segmentWidth)
	x_min = x - (segmentWidth/2)
	x_max = x + (segmentWidth/2)
	y_min = y - (segmentWidth/2)
	y_max = y + (segmentWidth/2)
	
	image[int(y_min):int(y_max),int(x_min):int(x_max)] = segment
	
	return image

def removeHotPixel(image,x,y,width):

	segment = imageSegment(imageArr,x,y,width)
	filtered = ndimage.median_filter(segment,size=width)
	newImage = replaceImageSegment(imageArr,filtered,x,y,width)
	return newImage









#newImage = removeHotPixel(imageArr,246,1548,10)
#newImage = removeHotPixel(newImage,152,1275,10)
#newImage = removeHotPixel(newImage,1580,788,10)
#newImage = removeHotPixel(newImage,855,1679,10)
#newImage = removeHotPixel(newImage,383,1285,10)
#newImage = removeHotPixel(newImage,125,167,10)
#newImage = removeHotPixel(newImage,1103,272,10)

#plt.imshow(imageSub)
#plt.show()
fig = plt.figure(figsize=(15, 15))

for i in range(0,1):
    file = "%03d" % (i)
    path = 'Images/Chamber_Flange_Tomo_'
    print(i,file)

    imageTif = openImage(path+file+".tif")
    imageArr = image2Numpy(imageTif)
    original = imageArr.copy()
    ax = fig.add_subplot(211)
    ax.imshow(original)
    plt.show()
    filter1 = imageArr < 1.1e4
    imageSub = imageArr.copy()
    imageSub[filter1] = 0
    newImage = imageArr.copy()

    for x in range(0,imageSub.shape[0]):
        for y in range(0,imageSub.shape[1]):
            if(imageSub[x,y]>0.1):
                temp = newImage.copy()
                newImage = removeHotPixel(newImage,y,x,10)
    ax = fig.add_subplot(212)
    ax.imshow(original-newImage)
    plt.show()
	#pathW = 'Cleaned/Chamber_Flange_Tomo_'
	#saveImage(newImage,pathW+file+".tif")
