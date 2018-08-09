from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import math
from keras.metrics import categorical_accuracy
from matplotlib.animation import FuncAnimation
from astroML.utils import completeness_contamination
from astroML.utils import split_samples
from scipy.fftpack import fft, ifft
from keras.models import load_model

#Convert labels from label to CNTK output format, basically an array of 0's with a 1 in the position of the desired label so 9 = [0 0 0 0 0 0 0 0 0 1]
def convertLabels(labels,samplesize,out):
    label = np.zeros((samplesize,out),dtype=np.float32)
    for i in range(0,len(labels)):
        if labels[i] == 0:
            label[i,:] = np.array([0,1])
        else:
            label[i,:] = np.array([1,0])
    return label

def reBalanceData(x,y,Multip):
    ones = x[np.where(y==1)].copy()
    y_ones = y[np.where(y==1)].copy()
    total = len(y)
    total_one = len(ones)
    multiplier = int(math.ceil((total/total_one)*Multip))
    for i in range(multiplier):
        x = np.insert(x,1,ones,axis=0)
        y = np.insert(y,1,y_ones,axis=0)

    ran = np.arange(x.shape[0])
    np.random.shuffle(ran)
    x= x[ran]
    y= y[ran]
    return x,y


def predictionMap(xlim,ylim):
    mesh = []
    for x in np.arange(xlim[0],xlim[1],0.001):
        for y in np.arange(ylim[0],ylim[1],0.001):
            mesh.append([x,y,x*x,y*y])     
    return (np.array(mesh))

def splitdata(X,y,ratio):
    length = X.shape[0]
    return X[:int(length*ratio)],X[:int(length*(1-ratio))],y[:int(length*ratio)],y[:int(length*(1-ratio))]


def generateData(multi):
    X = np.loadtxt('AstroML_Data.txt')
    y =  np.loadtxt('AstroML_Labels.txt')
    

    ran = np.arange(X.shape[0])
    np.random.shuffle(ran)
    X= X[ran]
    y= y[ran] 
    
    X_train, X_test, y_train, y_test = splitdata(X, y,multi)
    X_train, y_train = reBalanceData(X_train,y_train,1.0-multi)

    np.save('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy',X_train)    
    np.save('AstroML_X_Test_Shuffle_Split_0_7.npy',X_test)    
    np.save('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy',y_train)    
    np.save('AstroML_Y_Test_Shuffle_Split_0_7.npy',y_test)   
    
    X_test, y_test = reBalanceData(X_train,y_train,1.0-multi) 
    
    np.save('AstroML_X_Test_Shuffle_Split_0_7_Rebalance_1.npy.npy',X_test)    
    np.save('AstroML_Y_Test_Shuffle_Split_0_7_Rebalance_1.npy.npy',y_test)    

def addSquaredColumn(X_train,X_test,X_test_unbalanced):
	for i in range(0,4):
		X_train=np.append(X_train,np.multiply(X_train[:,[i]],X_train[:,[i]]),axis=1)
		X_test=np.append(X_test,np.multiply(X_test[:,[i]],X_test[:,[i]]),axis=1)
		X_test_unbalanced=np.append(X_test_unbalanced,np.multiply(X_test_unbalanced[:,[i]],X_test_unbalanced[:,[i]]),axis=1)
            
	return X_train,X_test,X_test_unbalanced
#%%
############################################
#GPU Checking
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
keras.backend.tensorflow_backend._get_available_gpus()

############# Settings #####################
network = [[4,"tanh"],[3,"tanh"],[1,"sigmoid"]]
LR = 0.1
Epochs = 100
BatchSize = int(math.ceil((np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy').shape[0])))
Multip = 0.7
addMulti = False
#Load old Model (True) Build new (False)
load=False
#############################################################
    
comp = []
cont = []
color = []

fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)

for coll in range(2,5):
    
	X = np.loadtxt('AstroML_Data.txt')
	y =  np.loadtxt('AstroML_Labels.txt')    

	X_train = np.load('AstroML_X_Train_Shuffle_Split_0_7_Rebalance_1.npy')
	X_test =  np.load('AstroML_X_Test_Shuffle_Split_0_7_Rebalance_1.npy')
	y_train = np.load('AstroML_Y_Train_Shuffle_Split_0_7_Rebalance_1.npy')
	y_test =  np.load('AstroML_Y_Test_Shuffle_Split_0_7_Rebalance_1.npy')
	X_test_unbalanced = np.load('AstroML_X_Test_Shuffle_Split_0_7.npy')
	y_test_unbalanced = np.load('AstroML_Y_Test_Shuffle_Split_0_7.npy')
      
	colSort = []
    
	if addMulti == True:
		X_train,X_test,X_test_unbalanced = addSquaredColumn(X_train,X_test,X_test_unbalanced)
		colSort2 = [1,0,5,4]
		colSort3 = [1,0,2,5,4,6]
		colSort4 = [1,0,2,3,5,4,6,7]
	else:
		colSort2 = [1,0]
		colSort3 = [1,0,2]
		colSort4 = [1,0,2,3]

	if coll==2:
		X_train = X_train[:, colSort2]  # rearrange columns for better 2-color results
		X_test = X_test[:, colSort2]  # rearrange columns for better 4-color results
		X = X[:, colSort2]
		X_test_unbalanced = X_test_unbalanced[:, colSort2]  # rearrange columns for better 2-color results
	elif coll==3:
		X_train = X_train[:, colSort3]  # rearrange columns for better 3-color results
		X_test = X_test[:, colSort3]  # rearrange columns for better 4-color results
		X = X[:, colSort3]
		X_test_unbalanced = X_test_unbalanced[:, colSort3]  # rearrange columns for better 2-color results
	elif coll==4:
		X_train = X_train[:, colSort4]  # rearrange columns for better 4-color results
		X_test = X_test[:, colSort4]  # rearrange columns for better 4-color results
		X = X[:, colSort4]
		X_test_unbalanced = X_test_unbalanced[:, colSort4]  # rearrange columns for better 2-color results

	N_tot = y_train.shape[0]
	#Total assignments of 0 (Classification = true)
	N_st = np.sum(y_train == 0)
	#Total assignments of 1 (Classification = false)
	N_rr = N_tot - N_st
	N_plot = 5000 + N_rr

	#%%

	###########################################################
	############Netowork Building##############################

	if load == True:
		if coll==1:
			model = load_model('model1.h5')
		elif coll==2:
			model = load_model('model2.h5')
		elif coll==3:
			model = load_model('model3.h5')
	else:
		layers = []
		layers.append(keras.layers.Dense(network[0][0],input_dim=(coll),kernel_initializer='normal', activation=network[0][1]))
		for layer in range(1,len(network)):      
            #Dropout
			if network[layer][0] == -1:
				layers.append(keras.layers.Dropout(network[layer][1]))	
			else:
				layers.append(keras.layers.Dense(network[layer][0],kernel_initializer='normal', activation=network[layer][1]))	   
		model = keras.Sequential(layers)


    ###############################
    #Training
    ###############################
    
	model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

	history = model.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size=BatchSize,epochs=Epochs, verbose=2)

	predictions = np.around(model.predict(X_test_unbalanced).reshape(model.predict(X_test_unbalanced).shape[0],))

	completeness, contamination = completeness_contamination(predictions,(y_test_unbalanced))
    
    ##############################
    #Model Evaluation
    ##############################
    
	scores = model.evaluate(X_test,y_test)
	loss = scores[0]
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
	comp.append(completeness)
	cont.append(contamination)
	color.append(coll)
    
	print("completeness",completeness)
	print("contamination", contamination)

	loss_data = history.history['loss']
	epoch_data = np.arange(0,len((loss_data)))
	crossVal = history.history['val_loss']
  
    ####################################
    #Loss Plotting
    ####################################
    
	ax_loss = fig.add_subplot(221)

	im_loss = ax_loss.plot(epoch_data,np.log(loss_data),'-',label=str(coll+1)+" Colours")
	ax_loss.plot(epoch_data,np.log(crossVal),'-',label=str(coll+1)+" Colours Cross Val")
	ax_loss.set_ylabel('Log(Loss)')
    	
	ax_loss.set_xlabel('Epoch')
	ax_loss.legend()
    
    #Save Model
	model.save('model'+str(coll)+'.h5')

	if coll == 2 :
		
		xlim = (0.7, 1.35)
		ylim = (-0.15, 0.4)
		
		#predictions = np.transpose(model.predict(X_train))

		test = predictionMap(xlim,ylim)
        
		xshape = int((xlim[1]-xlim[0])*1000)+1
		yshape = int((ylim[1]-ylim[0])*1000)
    
		test = test[:,colSort2]
		predictions =(model.predict(test))

		ax_heat = fig.add_subplot(222)

		im_heat = ax_heat.imshow(np.transpose(np.reshape(predictions[:,0],(xshape,yshape))),origin='lower',extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
		cb = fig.colorbar(im_heat, ax=ax_heat)
		cb.set_label('Classification Probability of Variable Main Sequence Stars.')
		ax_heat.set_xlabel('$u-g$')
		ax_heat.set_ylabel('$g-r$')

		ac_cont = fig.add_subplot(223)

		im_cont = ac_cont.scatter(X[-N_plot:, 1],X[-N_plot:, 0], c=y[-N_plot:],s=12, lw=0, cmap=plt.cm.binary, zorder=2)
		ac_cont.contour(np.reshape(test[:, 1],(xshape,yshape)), np.reshape(test[:, 0],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)),cmap=plt.cm.binary,lw=2)
		im_cont.set_clim(-0.5, 1)
		ac_cont.set_xlabel('$u-g$')
		ac_cont.set_ylabel('$g-r$')
		hiddenLayers = "Input: " + str(coll+1) + " "

		hiddenLayers = hiddenLayers + " "+str(network[0][0]) + " (activation = "+str(network[0][1])+") "
	
		for layer in range(1,len(network)):
			hiddenLayers = hiddenLayers +str(network[layer][0]) + " (activation = "+str(network[layer][1])+") "

		fig.suptitle("Epochs = "+str(Epochs)+" Batch Size = "+str(BatchSize)+", Multi = "+str(Multip)+", Learning Rate = "+str(LR)+ "\n Layers-> " +hiddenLayers +"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100),fontsize=22,y=0.98)


####################
#astroML Data
####################
#compML = np.array([0.68613139])
compML = np.array([0.68613139, 0.81021898, 0.87591241])
contML =  np.array([ 0.79295154, 0.80143113, 0.79020979])
#contML =  np.array([ 0.79295154])

ax = fig.add_subplot(224)
ax.plot(color, comp, 'o-r', ms=6,label="TensorFlow-Completeness")
ax.plot(color, compML, 'o-k', ms=6,label="Gaussian Naive Bayes-Completeness")
ax.plot(color, cont, 'ro--', ms=6,label="TensorFlow-Contamination")
ax.plot(color, contML, 'ko--', ms=6,label="Gaussian Naive Bayes-Contamination")
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.legend()
ax.set_ylabel('Completeness/Contamination')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.set_xlim([1.5,4.5])

plt.tight_layout()
plt.subplots_adjust(hspace = 0.2,wspace=0.2,top=0.89,bottom=0.05)
plt.show()







