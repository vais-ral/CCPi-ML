# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.utils.data
def loadMATData(file1):
    return sio.loadmat(file1)



def fit(model,optimizer,dataloader,loss_fn,epochs,batch):
    
    plot_data = {"Epoch":[],"Batch":[],"Loss":[],"DeltaLoss":[],"Speed":[]} 
    store = 0
    for t in range(epochs):
        start = time.time()
        running_loss = 0.0
        #running_corrects = 0.0
        count = 0
        for i, data in enumerate(dataloader,0):
            inputs,label = data
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
            y_pred = model(inputs)
        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss
        #torch.LongTensor(torch.max()) takes care of the fact we are using crossEntropy loss function and 
            loss = loss_fn(y_pred, torch.LongTensor(torch.max(label,1)[1]))
    
            # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
    
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
            loss.backward()
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        
            optimizer.step()
            _,prediction = torch.max(y_pred,1)
            running_loss += loss.item() 
            count+=1
            #running_corrects += torch.sum(prediction == label.data)
        stop = time.time()
        epoch_loss = running_loss / len(dataloader)
        plot_data["Epoch"].append(t+1)
        plot_data["Batch"].append(batch)
        if t == 0:
            plot_data["DeltaLoss"].append("Nan")
        else:
            plot_data["DeltaLoss"].append(float(plot_data["Loss"][-1])-epoch_loss)
        plot_data["Loss"].append(epoch_loss)
        plot_data["Speed"].append(stop-start)

        #epoch_acc = running_corrects.double() / i   
        
        #print('Epoch:',str(t+1)+'/'+str(epochs),'Time:',stop-start,'Loss:',(plot_data["Loss"][-1]),count,batch)
    
    return model,plot_data
    
def testModel(model,dataLoader,labels_test):
    counts = 0
    for i, data in enumerate(dataLoader,0):
        inputs,label = data
        y_pred = torch.argmax(model(inputs),dim=1)
        counts +=torch.sum((y_pred == torch.argmax(label,dim=1)),0)
    print("Counts:",counts)
    
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        assi = labels[i]
        if labels[i] == 10:
            assi = 0
        label[i][assi] = 1.0
    return label


def dataSplit(features,labels,trainSp,batch):
    
    feat_train = features[:int(features.shape[0]*trainSp)]
    labels_train = labels[:int(labels.shape[0]*trainSp)]
    
    feat_test = features[int(labels.shape[0]*trainSp):]
    labels_test = labels[int(labels.shape[0]*trainSp):]
    
    tensor_x = torch.stack([torch.Tensor(i) for i in feat_train]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in labels_train])
    
    my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = torch.utils.data.DataLoader(my_dataset,batch_size=batch) # create your dataloader
    
    tensor_x_test = torch.stack([torch.Tensor(i) for i in feat_test]) # transform to torch tensors
    tensor_y_test = torch.stack([torch.Tensor(i) for i in labels_test])
    
    my_dataset_test = torch.utils.data.TensorDataset(tensor_x_test,tensor_y_test) # create your datset
    my_dataloader_test = torch.utils.data.DataLoader(my_dataset_test,batch_size=batch) # create your dataloader

    return my_dataset,my_dataloader,my_dataset_test,my_dataloader_test

####### Network Topology ###########
 
### N = Batch number , D_in = input dimensions, H = didden layer size ,D_Out = output dimensions
##### BATCH COUNTING DOES NOT WORK YET!!!!! In testModel function y_pred outputs a tensor the size of BAtch size if you do torch.max(y_preds,1) you will get a tensor with the prediction for each of the items in the batch

Epochs = 100
Learning_rate=0.1
Momentum = 0.9

###### Input Data, Shuffle, Format into PyTorch Tensor ###########

mat_cont = sio.loadmat('ex3data1.mat')

features = mat_cont['X']
labels = mat_cont['y']

ran = np.arange(features.shape[0])
np.random.shuffle(ran)
features = features[ran]
labels = labels[ran]


filter1 = labels == 10
labels[filter1] = 0.0

for N in range(0,50,4):

    if N == 0:
        N=1
    D_in, H, D_out =  400, N, 10
    labels = convertLabels(labels,labels.shape[0],D_out)

    print('batch',N)
    train_dataset,train_dataloader,test_dataset,test_dataloader = dataSplit(features,labels,0.7,40)
####### Build Network Model ##########
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),

    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)
    
    
    ###### Train and Test ##########
    
    model, plot_data = fit(model,optimizer,train_dataloader,loss_fn,Epochs,40)
    
    #testModel(model,my_dataloader_test,labels)
#
    print(plot_data["Loss"][len(plot_data["Loss"])-1])
    plt.plot(plot_data["Epoch"], plot_data["Loss"], 'b-')
    
    plt.xlabel('Epoch')
    
    plt.ylabel('Loss')
    
    plt.title('Epoch vs. Training loss')
    
    plt.show()
#    
#    
    ########## Data Writing############
    f = open('Data/PyTorch_data_batchnum_'+str(N+1)+".txt","w")
    
    for i in range(0,len(plot_data["Epoch"])):
        f.write(str(plot_data["Epoch"][i])+","+str(N)+","+str(plot_data["Loss"][i])+","+str(plot_data["DeltaLoss"][i])+","+str(plot_data["Speed"][i])+"\n")
        
        
    
    f.close()

