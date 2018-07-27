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

#fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\cntk_data_batchnum_'
filePyTorch = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\PyTorch\Data\Vairable_BatchSize\PyTorch_data_batchnum_'
fileCNTK = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\CNTK\Data\cntk_data_batchnum_'
fileTensorFlow = r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML\Tensorflow\Benchmarking\Variable_Batchsize\TF_loss_data_batchnum_'

cntk_data = []
pytorch_data = []
tf_data = []

for file in range(0,500,10):
    
   # data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')    
    data_pytorch = np.genfromtxt(filePyTorch+str(file+1)+'.txt',delimiter=',')
    data_cntk = np.genfromtxt(fileCNTK+str(file+1)+'.txt',delimiter=',')
    data_tf = np.genfromtxt(fileTensorFlow+str(file+1)+'.txt',delimiter=',')
    print(file,data_pytorch)
    #cntk_data.append(np.transpose(data_cntk))
    pytorch_data.append(np.transpose(data_pytorch))
    cntk_data.append(np.transpose(data_cntk))
    tf_data.append(np.transpose(data_tf))

#############################################################################################################
#%%
fig, ax = plt.subplots()
count = 0

li= [50,500]
col = [['orange','b','g'],['r','m','k']]
for i in range(0,2):
    plotx = 0
    ploty = 1
    limit = li[i]
    columnSearch = 0
    limitLogic = 2
    logic = ['<=','>=','=']
    columnLabels = ['Epochs','BatchSize','Loss','DeltaLoss','Speed']
    Title = "For " + columnLabels[columnSearch] + ' ' + logic[limitLogic] + ' ' + str(limit)+' , '
    columnLabelsTf = ['Epochs','BatchSize','Loss','DeltaLoss','Speed']
    columnLabelsPy = ['Epochs','BatchSize','Loss','DeltaLoss','Speed']
    dl_pytorch, columnLabelspy = dataLimit(pytorch_data,limit,columnSearch,columnLabelsPy,2)
    dl_cntk, columnLabels2 = dataLimit(cntk_data,limit,columnSearch,columnLabels,2)
    dl_tf, columnLabels2 = dataLimit(tf_data,limit,columnSearch,columnLabelsTf,2)
       
    pyPol =poly.polyfit(dl_pytorch[plotx],dl_pytorch[ploty],4)
    pyCntk = poly.polyfit(dl_cntk[plotx],dl_cntk[ploty],4)
    pyTF = poly.polyfit(dl_tf[plotx],dl_tf[ploty],4)
       
       
    x = np.arange(0,dl_pytorch[plotx][-1],0.01)
       
    ffit = poly.polyval(x, pyPol)
    plt.plot(x, ffit,col[i][0],lw=4)
    ffit = poly.polyval(x, pyCntk)
    plt.plot(x, ffit,col[i][1],lw=4)
    ffit = poly.polyval(x, pyTF)
    plt.plot(x, ffit,col[i][2],lw=4)
    print(columnLabels2,dl_pytorch)
    plt.scatter(dl_cntk[plotx],dl_cntk[ploty],label="CNTK, Epoch = "+str(li[i]),s=40,c=col[i][1])
    plt.scatter(dl_tf[plotx],dl_tf[ploty],label="TensorFlow, Epoch = "+str(li[i]),s=40,c=col[i][2])
    plt.title('Loss at Epoch 50 and 500 for Various Batch Sizes',fontsize=20)
    plt.scatter(dl_pytorch[plotx],dl_pytorch[ploty],label="PyTorch, Epoch = "+str(li[i]),s=40,c=col[i][0])
    plt.legend(loc=2, prop={'size': 20})
    plt.tick_params(labelsize=20)
    plt.xlabel(columnLabels[plotx],fontsize=20)
    plt.ylabel(columnLabels[ploty],fontsize=20)
plt.show()