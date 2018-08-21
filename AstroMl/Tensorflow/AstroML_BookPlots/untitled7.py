# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:50:14 2018

@author: zyv57124
"""
import numpy as np
from astroML.datasets import fetch_LINEAR_geneva

data = fetch_LINEAR_geneva()
fname='jhhjj.npy'
np.save(fname,data)