# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:50:16 2018

@author: zyv57124
"""
import numpy
import PIL
from PIL import Image

y_max = 256
x_max = 256

arr = numpy.zeros((256, 256), dtype=numpy.uint8)
y_max, x_max = arr.shape

noise = numpy.random.normal(1, 1, (256, 256))
arr = arr + noise
print (arr)
im = Image.fromarray(arr, 'L')
im.save('./blah.png')
im.show()