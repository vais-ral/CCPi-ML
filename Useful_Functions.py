# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:00:30 2018

@author: zyv57124
"""


#Curved line of best fit on scatter graph
def PolyFit(x, y, z, n, x_values):    
    z = np.polyfit(x, y, n)
    x_fit = x_values  #Range along x axis length with evenly spaced intervals
    y_fit = z[0]*(x_fit**3) + z[1]*(x_fit**2) + z[2]*(x_fit**1) + z[3]*(x_fit**0)
    plt.plot(x_fit,y_fit,'k-')    #Fit is black line
    
    
#Timing
from time import time
class TimingCallback(keras.callbacks.Callback):
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)
#cb=TimingCallBack


def tt_split(Data, Labels, test_size, shuffle):
    X_train, X_test,y_train, y_test = train_test_split(Data_Astro, Labels_Astro,test_size=0.2, shuffle=True)
    return X_train, X_test,y_train, y_test