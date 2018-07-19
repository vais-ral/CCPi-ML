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
    del labels[column]
    
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

    return np.transpose(arr), labels

#fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\cntk_data_batchnum_'
filePyTorch = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\PyTorch\PyTorch_data_batchnum_'

cntk_data = []
pytorch_data = []

for file in range(0,500,10):
    
   # data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_pytorch = np.genfromtxt(filePyTorch+str(file+1)+'.txt',delimiter=',')
    
    #cntk_data.append(np.transpose(data_cntk))
    pytorch_data.append(np.transpose(data_pytorch))
    

plotx = 0
ploty = 1
limit = 200
columnSearch = 0
limitLogic = 2
logic = ['<=','>=','=']
columnLabels = ['Epochs','BatchSize','Loss','DeltaLoss','Speed']
Title = "For " + columnLabels[columnSearch] + ' ' + logic[limitLogic] + ' ' + str(limit)+' , '

dl_pytorch, columnLabels = dataLimit(pytorch_data,limit,columnSearch,columnLabels,2)
plt.title(Title+columnLabels[ploty] + ' Vs ' + columnLabels[plotx])

print(columnLabels,dl_pytorch)
plt.scatter(dl_pytorch[plotx],dl_pytorch[ploty])
plt.xlabel(columnLabels[plotx])
plt.ylabel(columnLabels[ploty])
plt.show()