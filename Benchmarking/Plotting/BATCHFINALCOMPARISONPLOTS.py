# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:42:33 2018

@author: zyv57124
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from array import array
import pylab
import scipy
from scipy import optimize

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


#fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\cntk_data_batchnum_'
fileKeras = r'C:\Users\zyv57124\Documents\GitHub\CCPi-ML\TensorFlow\Data\KER_loss_data_batchnum_'
filetensorflow = r'C:\Users\zyv57124\Documents\GitHub\CCPi-ML\TensorFlow\Data\TF_loss_data_batchnum_'
fileCNTK = r'C:\Users\zyv57124\Documents\GitHub\CCPi-ML\CNTK\Data\cntk_data_batchnum_'
filePyTorch = r'C:\Users\zyv57124\Documents\GitHub\CCPi-ML\PyTorch\Data\PyTorch_data_batchnum_'

Tensorflow_data = []
Keras_data = []
CNTK_data = []
PyTorch_data = []


##--------------------------------------------------------------------------------------------
##CHANGING BATCH SIZE
##--------------------------------------------------------------------------------------------

count = 0
for file in range(0,500,10):
    
   # data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_Tensorflow = np.genfromtxt(filetensorflow+str(file+1)+'.txt',delimiter=',')
    
    data_Keras = np.genfromtxt(fileKeras+str(file+1)+'.txt',delimiter=',')
    
    data_CNTK = np.genfromtxt(fileCNTK+str(file+1)+'.txt', delimiter = ',')
    data_CNTK = data_CNTK[:,[0,2,1,4,3]]
    
    data_PyTorch = np.genfromtxt(filePyTorch+str(file+1)+'.txt', delimiter = ',')
    data_PyTorch = data_PyTorch[:,[0,2,1,4,3]]
    
    Tensorflow_data.append(np.transpose(data_Tensorflow))
    Keras_data.append(np.transpose(data_Keras))
    CNTK_data.append(np.transpose(data_CNTK))
    PyTorch_data.append(np.transpose(data_PyTorch))

plotx = 1
ploty = 0
limit = 99
columnSearch = 0
limitLogic = 2
logic = ['<=','>=','=']

#for i in range(0,50):
    #plt.scatter(PyTorch_data[i][0],PyTorch_data[i][1])
    #plt.show()

columnLabels = ['Epoch','Loss','Batchsize','Time','Change in Loss']

Title = "For " + columnLabels[columnSearch] + ' ' + logic[limitLogic] + ' ' + str(limit)+' , '

dl_Tensorflow = dataLimit(Tensorflow_data, limit, columnSearch, columnLabels,2)
dl_keras = dataLimit(Keras_data, limit, columnSearch, columnLabels,2)
dl_CNTK = dataLimit(CNTK_data, limit, columnSearch, columnLabels,2)
dl_PyTorch = dataLimit(PyTorch_data, limit, columnSearch, columnLabels,2)

plt.title(Title+columnLabels[ploty+1] + ' Vs ' + columnLabels[plotx+1])

plt.scatter(dl_Tensorflow[plotx],dl_Tensorflow[ploty], label='Tensorflow')
z = np.polyfit(dl_Tensorflow[plotx],dl_Tensorflow[ploty], 3)
x_fit = np.arange(0,500,1)
y_fit = z[0]*(x_fit**3) + z[1]*(x_fit**2) + z[2]*(x_fit**1) + z[3]*(x_fit**0)
#plt.plot(x_fit,y_fit,'b-')

plt.scatter(dl_keras[plotx],dl_keras[ploty], label='Keras')
z = np.polyfit(dl_keras[plotx],dl_keras[ploty], 3)
x_fit = np.arange(0,500,1)
y_fit = z[0]*(x_fit**3) + z[1]*(x_fit**2) + z[2]*(x_fit**1) + z[3]*(x_fit**0)
#plt.plot(x_fit,y_fit,'r-')

plt.scatter(dl_CNTK[plotx], dl_CNTK[ploty], label='CNTK')
z = np.polyfit(dl_CNTK[plotx],dl_CNTK[ploty], 3)
x_fit = np.arange(0,500,1)
y_fit = z[0]*(x_fit**3) + z[1]*(x_fit**2) + z[2]*(x_fit**1) + z[3]*(x_fit**0)
#plt.plot(x_fit,y_fit,'g-')

plt.scatter(dl_PyTorch[plotx], dl_PyTorch[ploty], label='PyTorch')
z = np.polyfit(dl_PyTorch[plotx],dl_PyTorch[ploty], 3)
x_fit = np.arange(0,500,1)
y_fit = z[0]*(x_fit**3) + z[1]*(x_fit**2) + z[2]*(x_fit**1) + z[3]*(x_fit**0)
#plt.plot(x_fit,y_fit,'r-')

plt.xlabel(columnLabels[plotx+1])
plt.ylabel(columnLabels[ploty+1])
plt.ylim(0,1)
plt.legend()
plt.show()

