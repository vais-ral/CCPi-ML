# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:03:48 2018

@author: lhe39759
"""

import numpy as np
import sys
def findLimitIndex(arr,limit,limitLogic):
    if limitLogic <0 or limitLogic > 2:
        print("Invalid logic choice options: 0: <=, 1: >=, 2: ==")
        sys.exit()
    else:
 
        if limitLogic == 0:
            filter1 = arr<=limit
        elif limitLogic == 1:
            filter1 = arr>=limit
        elif limitLogic == 2:
            filter1 = arr==limit

        if np.any(filter1):
            return np.where(filter1)[0][0]
        else:
            print('Limit of:',limit,', did not find result in array',arr)
            return 'Stop'


def dataLimit(data,limit,column,labels,limitLogic):
    length = len(data)
    arr = np.zeros((length,4))
   
    
    for i in range(length):

        index = findLimitIndex(data[i][column],limit,limitLogic)
        if index != 'Stop':
            columnSearch = np.arange(0,5,1)
            columnSearch = np.delete(columnSearch,np.where(columnSearch==column)[0][0],0)  
            itter = 0
            
            for n in columnSearch:
                arr[i][itter] = data[i][n][index]
                itter +=1
        else:
            break

    return np.transpose(arr)
fileCNTK = r'C:\Users\zyv57124\Documents\Documents\GitHub\CCPi-ML\CNTK\Data\cntk_data_batchnum_'
fileKeras = r'C:\Users\zyv57124\Documents\Documents\GitHub\CCPi-ML\Keras\Data\KER_loss_data_batchnum_'
filetensorflow = r'C:\Users\zyv57124\Documents\Documents\GitHub\CCPi-ML\TensorFlow\Data\TF_loss_data_batchnum_'
filePyTorch = r'C:\Users\zyv57124\Documents\Documents\GitHub\CCPi-ML\PyTorch\Data\PyTorch_data_batchnum_'

Tensorflow_data = []
Keras_data = []
CNTK_data = []
PyTorch_data = []

count = 0
for file in range(0,500,10):
    
   # data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_Tensorflow = np.genfromtxt(filetensorflow+str(file+1)+'.txt',delimiter=',')
    data_Keras = np.genfromtxt(fileKeras+str(file+1)+'.txt',delimiter=',')
    data_CNTK = np.genfromtxt(fileCNTK+str(file+1)+'.txt', delimiter = ',')
    data_PyTorch = np.genfromtxt(filePyTorch+str(file+1)+'.txt', delimiter = ',')
    #cntk_data.append(np.transpose(data_cntk))
    Tensorflow_data.append(np.transpose(data_Tensorflow))
    Keras_data.append(np.transpose(data_Keras))
    CNTK_data = (np.transpose(data_CNTK))
    PyTorch_data = (np.transpose(data_PyTorch))

plotx = 0
ploty = 2
limit = 100
columnSearch = 0
limitLogic = 2
logic = ['<=','>=','=']

columnLabels = ['Epoch','Loss','Batchsize','Time','Change in Loss']

Title = "For " + columnLabels[columnSearch] + ' ' + logic[limitLogic] + ' ' + str(limit)+' , '

dl_Tensorflow = dataLimit(Tensorflow_data, limit, columnSearch, columnLabels,2)
dl_keras = dataLimit(Keras_data, limit, columnSearch, columnLabels,2)
dl_CNTK = dataLimit(CNTK_data, limit, columnSearch, columnLabels,2)
dl_PyTorch = dataLimit(PyTorch_data, limit, columnSearch, columnLabels,2)

plt.title(Title+columnLabels[ploty] + ' Vs ' + columnLabels[plotx])

plt.scatter(dl_Tensorflow[plotx],dl_Tensorflow[ploty])
plt.scatter(dl_keras[plotx],dl_keras[ploty])
plt.scatter(dl_CNTK[plotx], dl_CNTK[ploty])
plt.scatter(dl_PyTorch[plotx], dl_PyTorch[ploty])

plt.xlabel(columnLabels[plotx+1])
plt.ylabel(columnLabels[ploty+1])
plt.show()