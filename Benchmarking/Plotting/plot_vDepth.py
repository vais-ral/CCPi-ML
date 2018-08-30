# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:03:48 2018

@author: lhe39759
"""
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

import numpy as np
import sys
def findLimitIndex(arr,limit,limitLogic):
    
    if limitLogic <0 or limitLogic > 2:
        print("Invalid logic choice options: 0: <=, 1: >=, 2: ==")
        sys.exit()
    else:
        if limitLogic == 0:
            filter1 = np.abs(arr)<=limit
        elif limitLogic == 1:
            filter1 = np.abs(arr)>=limit
        elif limitLogic == 2:
            filter1 = arr==limit

        if np.any(filter1):
            return np.where(filter1)[0][0]
        else:
            print('Limit of:',limit,', did not find result in array',np.abs(arr))
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
def sumTime(data):
    out = np.zeros((len(data)))
    for i in range(0,len(data)):
        out[i] = np.sum(data[i][4])
    print(out)
    return out
        
#fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\cntk_data_batchnum_'
filePyTorch = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\PyTorch\Data\VDepth_12\PyTorch_data_batchnum_'
fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\Data\VDepth_12\cntk_data_batchnum_'
fileTensorFlow = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\Tensorflow\Benchmarking\VDepth_12\TF_loss_data_batchnum_'

cntk_data = []
pytorch_data = []
tf_data = []

for file in range(0,50,4):
    s = file
    d = file
    if file ==0:
        file =1
        
   # data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_pytorch = np.genfromtxt(filePyTorch+str(d)+'.txt',delimiter=',')
    data_cntk = np.genfromtxt(fileCNTK+str(file)+'.txt',delimiter=',')
    data_tf = np.genfromtxt(fileTensorFlow+str(s+1)+'.txt',delimiter=',')
    
    print(file,data_pytorch)
    #cntk_data.append(np.transpose(data_cntk))
    pytorch_data.append(np.transpose(data_pytorch))
    cntk_data.append(np.transpose(data_cntk))
    tf_data.append(np.transpose(data_tf))

#############################################################################################################
#%%
fig, ax = plt.subplots()
count = 0

col = [['orange','b','g'],['r','m','k']]
tf_time = sumTime(tf_data)
py_time = sumTime(pytorch_data)
cntk_time = sumTime(cntk_data)

widtharr = np.arange(0,50,4)



pyPol =poly.polyfit(widtharr,py_time,2)
pyCntk = poly.polyfit(widtharr,cntk_time,2)
pyTF = poly.polyfit(widtharr,tf_time,2)
   
x = np.arange(0,50,0.01)
print(x)
ffit = poly.polyval(x, pyPol)
plt.plot(x, ffit,col[0][0],lw=4)
ffit = poly.polyval(x, pyCntk)
plt.plot(x, ffit,col[0][1],lw=4)
ffit = poly.polyval(x, pyTF)
plt.plot(x, ffit,col[0][2],lw=4)

print(columnLabels2,dl_cntk,widtharr)

plt.scatter(widtharr,cntk_time,label="CNTK",s=80,c=col[0][1])
plt.scatter(widtharr,tf_time,label="TensorFlow",s=80,c=col[0][2])
plt.scatter(widtharr,py_time,label="PyTorch",s=80,c=col[0][0])

plt.title('Time Taken to Reach 200th Epoch for Varying Network Depth Each with 12 Artificial Neurons',fontsize=20)
plt.legend(loc=2, prop={'size': 20})
plt.tick_params(labelsize=20)
plt.xlabel("Network Depth",fontsize=20)
plt.ylabel("Time (s)",fontsize=20)
plt.show()