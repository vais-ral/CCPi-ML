# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.naive_bayes import GaussianNB
from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
from astroML.utils import completeness_contamination

#----------------------------------------------------------------------
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

#----------------------------------------------------------------------
# get data and split into training & testing sets
X, y = fetch_rrlyrae_combined()
X = X[:, [1, 0, 2, 3]]  # rearrange columns for better 1-color results
(X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.75, 0.25],
                                                     random_state=0)
#total test data
N_tot = len(y)
#Total assignments of 0 (Classification = true)
N_st = np.sum(y == 0)
#Total assignments of 1 (Classification = false)
N_rr = N_tot - N_st
#Number of train labels
N_train = len(y_train)
#Number of test labels
N_test = len(y_test)

N_plot = 5000 + N_rr

#----------------------------------------------------------------------
# perform Naive Bayes
#Empty classifier and prediction lists
classifiers = []
predictions = []
#NP array from 1 to size of x features (number of color spaces) 
Ncolors = np.arange(1, X.shape[1] + 1)

#Order in which the colors appar after previous rearrangment
order = np.array([1, 0, 2, 3])

#Production prodcition for increasing number of colors i.e first prediction based off one color, second off two colors and so on
for nc in Ncolors:
    #Initialise GaussianNB object
    clf = GaussianNB()
    #Train using specified number of colors from Ncolors array, input training data and corresponding number of features and labels
    clf.fit(X_train[:, :nc], y_train)
    #Resulting prediction based off test set of features
    y_pred = clf.predict(X_test[:, :nc])
    
    #Add classifier object and predictions to list
    classifiers.append(clf)
    predictions.append(y_pred)

#Use atroML completeness function to produce completeness and contamination
completeness, contamination = completeness_contamination(predictions, y_test)

print "completeness", completeness
print "contamination", contamination

#------------------------------------------------------------
# Compute the decision boundary

clf = classifiers[1]
xlim = (0.7, 1.35)
ylim = (-0.15, 0.4)

#Produce 2-D array from xlim and ylim
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 81),
                     np.linspace(ylim[0], ylim[1], 71))

#Produce probability from previously generated 2-D array based on previous classification to be plotted as heatmap 
Z = clf.predict_proba(np.c_[yy.ravel(), xx.ravel()])
#Reshape the resulting prediction the same as xx 2-D array
Z = Z[:, 1].reshape(xx.shape)

#----------------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(7, 2.5))
#Plot Size
fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.0,
                    left=0.1, right=0.95, wspace=0.2)

# left plot: data and decision boundary
#Plot position
ax = fig.add_subplot(121)
#Scatter plot of original data with colours according to original labels
im = ax.scatter(X[-N_plot:, 1], X[-N_plot:, 0], c=y[-N_plot:],
                s=4, lw=0, cmap=plt.cm.binary, zorder=2)
im.set_clim(-0.5, 1)

#Plot heatmap of prediction array z
im = ax.imshow(Z, origin='lower', aspect='auto',
               cmap=plt.cm.binary, zorder=1,
               extent=xlim + ylim)
im.set_clim(0, 1.5)
#plot decision boundary
ax.contour(xx, yy, Z, [0.5], colors='k')
#Plot limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)
#Plot labels
ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')

# Plot completeness vs Ncolors
ax = plt.subplot(222)
ax.plot(Ncolors, completeness, 'o-k', ms=6)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.set_ylabel('completeness')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)

# Plot contamination vs Ncolors
ax = plt.subplot(224)
ax.plot(Ncolors, contamination, 'o-k', ms=6)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
#Plot labels and limits
ax.set_xlabel('N colors')
ax.set_ylabel('contamination')
ax.set_xlim(0.5, 4.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(True)

plt.show()