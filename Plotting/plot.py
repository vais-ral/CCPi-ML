# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:03:48 2018

@author: lhe39759
"""

import numpy as np

def findDeltaLossLimitIndex(arr,limit):
    return np.where(arr<=limit)[0][0]

def deltaLossLimit(data,limit):
    length = len(data)
    arr = np.zeros((length,4))
    
    for i in range(length):
        #data[i][3][1:] 1: misses out nan in delta loss
        index = findDeltaLossLimitIndex(data[i][3][1:],0.1)
        #Epoch
        arr[i][0] = data[i][0][index]
        #Batch Size
        arr[i][1] = data[i][1][index]
        #Loss
        arr[i][2] = data[i][2][index]
        #Speed
        arr[i][3] = data[i][4][index]

    return np.transpose(arr)

fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\cntk_data_batchnum_'
filePyTorch = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\PyTorch\PyTorch_data_batchnum_'

cntk_data = []
pytorch_data = []

for file in range(0,500,10):
    
    data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_pytorch = np.genfromtxt(filePyTorch+str(file+1)+'.txt',delimiter=',')
    
    cntk_data.append(np.transpose(data_cntk))
    pytorch_data.append(np.transpose(data_pytorch))
    

dl_pytorch = deltaLossLimit(pytorch_data,0.001)
print(dl_pytorch[1],dl_pytorch[2])
plt.scatter(dl_pytorch[1],dl_pytorch[2])
plt.show()