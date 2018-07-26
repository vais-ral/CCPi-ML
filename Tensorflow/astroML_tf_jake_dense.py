from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.keras.metrics import categorical_accuracy
from matplotlib.animation import FuncAnimation
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

def reBalanceData(x,y):
    ones = x[np.where(y==1)].copy()
    y_ones = y[np.where(y==1)].copy()
    total = len(y)
    total_one = len(ones)
    multiplier = math.ceil(total/total_one)
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
#############################################################

#############Data Loading & Conversion######################
    
X = np.loadtxt('AstroML_Data.txt',dtype=float)
y =  np.loadtxt('AstroML_Labels.txt',dtype=float)

X = X[:, [1, 0]]  # rearrange columns for better 1-color results

## Shuffle Data
ran = np.arange(X.shape[0])
np.random.shuffle(ran)
X= X[ran]
y= y[ran]

X_re,y_re = reBalanceData(X,y)
X_train,X_test,y_train,y_test = splitdata(X_re,y_re,0.7)

#X_train, y_train = X, y

N_tot = y_train.shape[0]
#Total assignments of 0 (Classification = true)
N_st = np.sum(y_train == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st

N_plot = 5000 + N_rr

######## Plot Feature Data #########################

fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2,
                    left=0.1, right=0.95, wspace=0.2)
#Plot Size
fig.suptitle("Neural Network Classification of RR Layrae Stars using TensorFlow")
ax = fig.add_subplot(221)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
                s=12, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)

ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')

#%%

###########################################################
############Netowork Building##############################

class_weight = {0:1.,1:((N_tot/N_rr)*1.2)}

model = keras.Sequential()
model.add(keras.layers.Dense(6, input_dim=2, kernel_initializer='normal', activation='tanh'))
model.add(keras.layers.Dense(35,  kernel_initializer='normal', activation='tanh'))
model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

history = model.fit(X_train, y_train, batch_size=1500,epochs=100, verbose=2)
#history = model.fit(X_train, y_train, batch_size=1500,epochs=100, verbose=2,class_weight=class_weight)

######Loss Plotting################
loss_data = history.history['loss']
epoch_data = np.arange(0,len(loss_data))

ax_loss = fig.add_subplot(222)
im_loss = ax_loss.plot(epoch_data,loss_data,'k-')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
##############Evaluate the model ###################
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#%%




#%%

xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)
#zlim = (np.amin(X[:,[2]]),np.amax(X[:,[2]]))
#glim = (np.amin(X[:,[3]]),np.amax(X[:,[3]]))

predictions = np.transpose(model.predict(X_train))

test = predictionMap(xlim,ylim)

xshape = int((xlim[1]-xlim[0])*1000)+1
yshape = int((ylim[1]-ylim[0])*1000)

predictions =(model.predict(test[:,[1,0]]))

#%%
ax_heat = fig.add_subplot(223)

im_heat = ax_heat.imshow(np.transpose(np.reshape(predictions[:,0],(xshape,yshape))),origin='lower',extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
cb = fig.colorbar(im_heat, ax=ax_heat)
cb.set_label('Classification Probability of Variable Main Sequence Stars',fontsize=8)
ax_heat.set_xlabel('$u-g$')
ax_heat.set_ylabel('$g-r$')
print(predictions[predictions<0.2].shape,predictions[predictions>0.8].shape)

#%%

ac_cont = fig.add_subplot(224)

im_cont = ac_cont.scatter(X[-N_plot:, 1],X[-N_plot:, 0], c=y[-N_plot:],s=12, lw=0, cmap=plt.cm.binary, zorder=2)
ac_cont.contour(np.reshape(test[:, 0],(xshape,yshape)), np.reshape(test[:, 1],(xshape,yshape)), np.reshape(predictions,(xshape,yshape)),cmap=plt.cm.binary,lw=2)
im_cont.set_clim(-0.5, 1)
ac_cont.set_xlabel('$u-g$')
ac_cont.set_ylabel('$g-r$')

plt.show()






