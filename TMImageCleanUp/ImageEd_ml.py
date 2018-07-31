


from skimage.filters.rank import median
from PIL import Image as tifIm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import math
def openImage(path):
	return tifIm.open(path)

def saveImage(array,path):
	im = tifIm.fromarray(array) # float32
	im.save(path, "TIFF")

def image2Numpy(im):
	return np.array(im).astype(np.int16)

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
    medImage = ndimage.median_filter(imageArr.copy(),size=10)
    subImage = np.subtract(image,medImage)
    filterZero = subImage < 0
    subImage[filterZero] = 0

    index = np.argwhere(subImage > threshold)
    return index

fig = plt.figure(figsize=(15, 15))


file = "%03d" % (1)
path = 'Images/Chamber_Flange_Tomo_'
print(1,file)

imageTif = openImage(path+file+".tif")

imageArr = image2Numpy(imageTif)
ax = fig.add_subplot(211)
ax.imshow(imageArr)
index = findZingers(imageArr.copy(),3*10**3)
newImage = imageArr.copy()

for i in index:

    temp = newImage.copy()
    print(i[1],i[0])
    newImage = removeHotPixel(newImage,i[1],i[0],10)
    
print(imageArr.shape)
print(imageArr)  
ax = fig.add_subplot(212)
ax.imshow(newImage)
plt.show()
pathW = 'Chamber_Flange_Tomo_'
saveImage(newImage,pathW+file+".tif")
#print(imageArr.shape[0])
#origianl = imageArr.copy().astype(np.int16)
#print(imageArr)
#ax = fig.add_subplot(221)
#ax.imshow(imageArr)
#medImage = ndimage.median_filter(imageArr.copy(),size=10)
#
##medImage = ndimage.gaussian_filter(medImage,sigma=5)
#ax = fig.add_subplot(222)
#ax.imshow(medImage)
#num = 1002
#print(origianl[num][num],medImage[num][num],np.subtract(origianl,medImage)[num][num])
#sub = np.subtract(origianl,medImage)
#filter1 = sub <0
#sub[filter1] = 0
#ax = fig.add_subplot(223)
#ax.imshow(sub)
#
#plt.show()
#
#submed = ndimage.gaussian_filter(sub.copy(),sigma=3)
#ax = fig.add_subplot(224)
#ax.imshow(submed)
#
#plt.show()
