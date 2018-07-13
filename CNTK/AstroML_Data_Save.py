# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:11:09 2018

@author: lhe39759
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:21:16 2018

@author: lhe39759
"""
import numpy as np



from tempfile import TemporaryFile

from astroML.datasets import fetch_rrlyrae_combined
from astroML.utils import split_samples
from astroML.utils import completeness_contamination

#----------------------------------------------------------------------
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)



#############################################################

#############Data Loading & Conversion######################

X, y = fetch_rrlyrae_combined()

np.savetxt('AstroML_Data.txt',X)
np.savetxt('AstroML_Labels.txt',y)

print("Done")