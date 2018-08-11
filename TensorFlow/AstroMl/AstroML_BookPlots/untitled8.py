# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:30:22 2018

@author: zyv57124
"""

import numpy as np
from astroML.datasets import fetch_LINEAR_geneva
from astroML.datasets import fetch_dr7_quasar
from astroML.datasets import fetch_sdss_sspp

quasars = fetch_dr7_quasar()
stars = fetch_sdss_sspp()

np.save('quasars.npy', quasars)
np.save('stars.npy', stars)