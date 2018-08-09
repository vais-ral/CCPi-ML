# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:05:24 2018

@author: lhe39759
"""
import numpy as np
import matplotlib.pyplot as plt
c,w,d,ltest,ltrain,comp,cont = np.genfromtxt(r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\TensorFlow\AstroMl\WidthDepthData\4_data.txt',delimiter=',',unpack=True)

#reshape(depth,width)
filter1 = c == 2
fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
ax_loss = fig.add_subplot(121)
measure = (1-ltrain[filter1])*(ltest[filter1]/ltrain[filter1])
print(measure)
im_heat=ax_loss.imshow(measure.reshape(10,20))
cb = fig.colorbar(im_heat, ax=ax_loss)
ax_loss.set_xlabel('Width')
ax_loss.set_ylabel('Depth')


loss = np.load(r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\TensorFlow\AstroMl\WidthDepthData\loss20_4.npy')

print(loss.shape)
ax_loss = fig.add_subplot(122)

ax_loss.plot(loss[0],loss[1])
plt.show()