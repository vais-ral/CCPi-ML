# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:42:14 2018

@author: lhe39759
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.axes()

def addOutLayer(layer,num,text):
    
    radius = 1.0
    yaxis = 2.0 + radius
    xaxis = 10.0 +radius
    fontsize=10
    height = (yaxis)*float(num-1)
    
    for neu in range(0,num):
        circle = plt.Circle((xaxis*layer,(neu*yaxis)-(height/2.0)), radius=radius, fill=False,color='k', linewidth=0.3)
        plt.gca().add_patch(circle)
        plt.annotate('',xy=(xaxis*(layer+0.4),(neu*yaxis)-(height/2.0)),xytext=(xaxis*layer+radius,(neu*yaxis)-(height/2.0)), arrowprops=dict(facecolor='black', shrink=0.0,width=0.01,headwidth=4,headlength=4))
        circleClear = plt.Circle((xaxis*(layer+0.4),(neu*yaxis)-(height/2.0)), radius=radius, fill=False,color='k', linewidth=0.0)
        plt.gca().add_patch(circleClear)

        
    plt.text((xaxis*layer)-((fontsize/10)*len(text)/2.0),(height/2.0)+yaxis,text,fontsize=fontsize)
    print(len(text)/2.0)


def addLayer(layer,num,num2,text):
    
    radius = 1.0
    yaxis = 2.0 + radius
    xaxis = 10.0 +radius
    fontsize=10

    height = (yaxis)*float(num-1)
    
    for neu in range(0,num):
        
        for l2 in range(0,num2):
            layer2 = layer+1
            height2 = (yaxis)*float(num2-1)
            plt.annotate('',xy=(xaxis*layer2-radius,(l2*yaxis)-(height2/2.0)),xytext=(xaxis*layer+radius,(neu*yaxis)-(height/2.0)), arrowprops=dict(facecolor='black', shrink=0.0,width=0.01,headwidth=0.1,headlength=0.1))

        circle = plt.Circle((xaxis*layer,(neu*yaxis)-(height/2.0)), radius=radius, fill=False,color='k', linewidth=0.3)
        plt.gca().add_patch(circle)
        
    plt.text((xaxis*layer)-((fontsize/10)*len(text)/2.0),(height/2.0)+yaxis,text,fontsize=fontsize)

network = [[3,"Input"],[10,"tanh"],[10,'DIck head'],[5,"Out"]]

for layer in range(0,len(network)):
    
    if layer == len(network)-1:
        addOutLayer(layer,network[layer][0],network[layer][1])
    else:
        addLayer(layer,network[layer][0],network[layer+1][0],network[layer][1])
    
#plt.gca().add_patch(text)

plt.axis('off')
plt.axis('scaled')

plt.show()