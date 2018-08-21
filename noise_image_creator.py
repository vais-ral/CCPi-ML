# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:50:16 2018

@author: zyv57124
"""
import numpy
import PIL
from PIL import Image

arr = numpy.zeros((112, 112), dtype=numpy.uint8)
noise = numpy.random.normal(0.5, 0.5, (112, 112))
arr = arr + noise
print (arr)
im = Image.fromarray(arr, 'L')
im.save('./blah2.png')
im.show()