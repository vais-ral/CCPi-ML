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
            mesh.append([x,y,0,0])     
    return (np.array(mesh))

def splitdata(X,y,ratio):
    length = X.shape[0]
    return X[:int(length*ratio)],X[:int(length*(1-ratio))],y[:int(length*ratio)],y[:int(length*(1-ratio))]

#%%
############################################

############# Settings #####################

network = [[44,"tanh"],[26,"tanh"],[-1,0.15],[1,"sigmoid"]]
LR = 0.003
Epochs = 2000
BatchSize = 400
Multip = 0.4

#############################################################

#############Data Loading & Conversion######################
    
comp = []
cont = []
color = []

fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,left=0.1, right=0.95, wspace=0.2)
ax_loss = fig.add_subplot(232)

for coll in range(3,4):
    
	X = np.loadtxt('AstroML_Data.txt',dtype=float)
	y =  np.loadtxt('AstroML_Labels.txt',dtype=float)

	if coll == 0:
		X = X[:, [1]]  # rearrange columns for better 1-color results
	elif coll==1:
		X = X[:, [1,0]]  # rearrange columns for better 2-color results
	elif coll==2:
		X = X[:, [1,0,2]]  # rearrange columns for better 3-color results
	elif coll==3:
		X = X[:, [1,0,2,3]]  # rearrange columns for better 4-color results

	## Shuffle Data

	ran = np.arange(X.shape[0])
	np.random.shuffle(ran)
	X= X[ran]
	y= y[ran]


	#X_train,X_test,y_train,y_test = splitdata(X,y,0.7)
	(X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.75, 0.25],
		                                             random_state=0)
	X_train,y_train = reBalanceData(X_train,y_train,Multip)
	#X_train, y_train = X, y

	N_tot = y_train.shape[0]
	#Total assignments of 0 (Classification = true)
	N_st = np.sum(y_train == 0)
	#Total assignments of 1 (Classification = false)
	N_rr = N_tot - N_st

	N_plot = 5000 + N_rr


	#%%

	###########################################################
	############Netowork Building##############################

	class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

	layers = []
	layers.append(keras.layers.Dense(network[0][0],input_dim=(coll+1),kernel_initializer='normal', activation=network[0][1]))
	
	for layer in range(1,len(network)):
        
        #Dropout
		if network[layer][0] == -1:
			layers.append(keras.layers.Dropout(network[layer][1]))	
		else:
			layers.append(keras.layers.Dense(network[layer][0],kernel_initializer='normal', activation=network[layer][1]))	

	model = keras.Sequential(layers)

	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LR), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

	history = model.fit(X_train, y_train, batch_size=BatchSize,epochs=Epochs, verbose=2)

	predictions = np.around(model.predict(X_test).reshape(model.predict(X_test).shape[0],))

	completeness, contamination = completeness_contamination(predictions,(y_test))

	comp.append(completeness)
	cont.append(contamination)
	color.append(coll+1)

	print("completeness",completeness)
	print("contamination", contamination)

	loss_data = history.history['loss']
	epoch_data = np.arange(0,len((loss_data)))

	im_loss = ax_loss.plot(epoch_data,np.log(loss_data),'-',label=str(coll+1)+" Colours")
	ax_loss.set_xlabel('Epoch')
	ax_loss.legend()

	if coll == 1 :
		
		######## Plot Feature Data #########################

		#Plot Size
		ax = fig.add_subplot(231)
		#Scatter plot of original data with colours according to original labels
		im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
				s=12, lw=0, cmap=plt.cm.binary, zorder=2)
		im.set_clim(-0.5, 1)

		ax.set_xlabel('$u-g$')
		ax.set_ylabel('$g-r$')

		######Loss Plotting################
		
		ax_loss.set_ylabel('Log(Loss)')
		##############Evaluate the model ###################
		scores = model.evaluate(X_test, y_test)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


		#%%

		xlim = (0.7, 1.35)
		ylim = (-0.15, 0.4)
		
		predictions = np.transpose(model.predict(X_train))

		test = predictionMap(xlim,ylim)

		xshape = int((xlim[1]-xlim[0])*1000)+1
		yshape = int((ylim[1]-ylim[0])*1000)
		
		test = test[:,[1,0]]
		#test = np.append(test, np.multiply(test[:, [0]],test[:, [0]]), axis=1)
		#test = np.append(test, np.multiply(test[:, [1]],test[:, [1]]), axis=1)
		predictions =(model.predict(test))

		#%%
		ax_heat = fig.add_subplot(234)

		im_heat = ax_heat.imshow(np.transpose(np.reshape(predictions[:,0],(xshape,yshape))),origin='lower',extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
		cb = fig.colorbar(im_heat, ax=ax_heat)
		cb.set_label('Classification Probability of Variable Main Sequence Stars.')
		ax_heat.set_xlabel('$u-g$')
		ax_heat.set_ylabel('$g-r$')
		print(predictions[predictions<0.2].shape,predictions[predictions>0.8].shape)

		#%%

		ac_cont = fig.add_subplot(235)

		im_cont = ac_cont.scatter(X[-N_plot:, 1],X[-N_plot:, 0], c=y[-N_plot:],s=12, lw=0, cmap=plt.cm.binary, zorder=2)
		ac_cont.contour(np.reshape(test[:, 1],(xshape,yshape)), np.reshape(test[:, 0],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)),cmap=plt.cm.binary,lw=2)
		im_cont.set_clim(-0.5, 1)
		ac_cont.set_xlabel('$u-g$')
		ac_cont.set_ylabel('$g-r$')
		hiddenLayers = "Input: " + str(coll+1) + " "

		hiddenLayers = hiddenLayers + " "+str(network[0][0]) + " (activation = "+str(network[0][1])+") "
	
		for layer in range(1,len(network)):
			hiddenLayers = hiddenLayers +str(network[layer][0]) + " (activation = "+str(network[layer][1])+") "
		print(len(network),hiddenLayers)

		fig.suptitle("Epochs = "+str(Epochs)+" Batch Size = "+str(BatchSize)+", Multi = "+str(Multip)+", Learning Rate = "+str(LR)+ "\n Layers-> " +hiddenLayers +"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100),fontsize=22,y=0.98)


########
#astroML Data
#######
#0.48175182
compML = np.array([0.68613139])
#compML = np.array([0.68613139, 0.81021898, 0.87591241])
#0.85201794,
#contML =  np.array([ 0.79295154, 0.80143113, 0.79020979])
contML =  np.array([ 0.79295154])


ax = fig.add_subplot(233)
ax.plot(color, comp, 'o-r', ms=6,label="TensorFlow")
ax.plot(color, compML, 'o-k', ms=6,label="Gaussian Naive Bayes")
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.legend()
ax.set_ylabel('completeness')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.set_xlim([1.5,4.5])
# Plot contamination vs Ncolors
ax = fig.add_subplot(236)
ax.plot(color, cont, 'o-r', ms=6,label="TensorFlow")
ax.plot(color, contML, 'o-k', ms=6,label="Gaussian Naive Bayes")
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
#Plot labels and limits
ax.set_xlabel('N colours')
ax.set_ylabel('contamination')
ax.set_xlim(0.5, 4.5)
ax.legend()
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.set_xlim([1.5,4.5])
plt.tight_layout()
plt.subplots_adjust(hspace = 0.2,wspace=0.2,top=0.89,bottom=0.05)
plt.show()







