# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:05:24 2018

@author: lhe39759
"""
import numpy as np
import matplotlib.pyplot as plt
w,d,ltest,ltrain,comp,cont = np.genfromtxt(r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\TensorFlow\AstroMl\WidthDepthData\data.txt',delimiter=',',unpack=True)

#reshape(depth,width)
measure = (1-ltrain)*(ltest/ltrain)
print(measure)
plt.imshow(measure.reshape(6,6))
plt.xlabel('Width')
plt.ylabel('Depth')
plt.colorbar()
plt.show()