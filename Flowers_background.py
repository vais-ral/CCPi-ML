# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:44:31 2018

@author: zyv57124
"""

import numpy
import PIL
from PIL import Image

flowers = numpy.asarray(Image.open('flowers.png'))

background = Image.open('flowers.png')
#im.save('./blah2.png')
im.show()