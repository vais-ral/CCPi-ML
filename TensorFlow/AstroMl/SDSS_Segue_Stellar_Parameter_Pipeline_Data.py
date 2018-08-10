# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:13:48 2018

@author: zyv57124
"""

from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_sspp
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=False)

#------------------------------------------------------------
# Fetch the data
data = fetch_sdss_sspp()

# select the first 10000 points
data = data[:10000]

# do some reasonable magnitude cuts
rpsf = data['rpsf']
data = data[(rpsf > 15) & (rpsf < 19)]

# get the desired data
logg = data['logg']
Teff = data['Teff']

print(logg[:40], Teff[:40])

#------------------------------------------------------------
# Plot the data
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(Teff, logg, marker='.', markersize=2, linestyle='none', color='black')

ax.set_xlim(8000, 4500)
ax.set_ylim(5.1, 1)

ax.set_xlabel(r'$\mathrm{T_{eff}\ (K)}$')
ax.set_ylabel(r'$\mathrm{log_{10}[g / (cm/s^2)]}$')

plt.show()