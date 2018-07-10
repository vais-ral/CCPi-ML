# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:53:32 2018

@author: zyv57124
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:03:08 2018

@author: zyv57124
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB    #Scikit learn has estimwator to perform gaussian naive bayes classification
import astroML.plotting

X = np.random.random((100, 2))  #random array of 100 points, 2 dimensions
Y = (X[:,0] + X[:,1] > 1).astype(int)   #Y is array of 1's and 0's. 1 if coordinates added > 1. 0 if <1

gnb = GaussianNB()
gnb.fit(X, Y)
y_pred = gnb.predict(X)
x = np.transpose(X)    #Makes X dimensions (100x2)

# predict the classification probabilities on a grid

xlim = (0, 1)     # axes lengths
ylim = (0, 1)

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),    #meshgrid ??????
                     np.linspace(ylim[0], ylim[1], 81))

Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])         #predicts ??????
Z = Z[:, 1].reshape(xx.shape)         #Takes second column of Z and makes it same shape as xx meshgrid

fig = plt.figure(figsize=(5, 3.75))    #Creates figure
ax = fig.add_subplot(111)            #Subplot and position in figure
ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=2)   #Plots the array of 100 points. colour is black if Y=1
ax.contour(xx, yy, Z, [0.5], colors='k')    #Plots the decision boundary
plt.scatter(x[0], x[1])


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


plt.show()