#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

lf = pd.read_csv('asymm_lattice_functions.csv')

f, ax = plt.subplots(2, 1)
f.suptitle('Asymmetric lattice functions')
ax[0].plot(lf['s'], lf['beta_x'], label='beta x')
ax[0].legend(loc='best')
ax[0].set_ylabel('beta y [m]')

ax[1].plot(lf['s'], lf['beta_y'], label='beta y')
ax[1].legend(loc='best')
ax[1].set_ylabel('beta y [m]')
ax[1].set_xlabel('s [m]')

f, ax = plt.subplots(2, 1)
f.suptitle('Asymmetric lattice dispersions (x)')
ax[0].plot(lf['s'], lf['disp_x'], label='dispersion x')
ax[0].legend(loc='best')
ax[0].set_ylabel('dispersion (x) [m]')

ax[1].plot(lf['s'], lf['disp_px'], label='dispersion px')
ax[1].legend(loc='best')
ax[1].set_ylabel('dispersion (px) [m/m]')
ax[1].set_xlabel('s [m]')

f, ax = plt.subplots(2, 1)
f.suptitle('Asymmetric lattice phase')
ax[0].plot(lf['s'], lf['psi_x'], label='psi x')
ax[0].legend(loc='best')
ax[0].set_ylabel('psi x')

ax[1].plot(lf['s'], lf['psi_y'], label='psi y')
ax[1].legend(loc='best')
ax[1].set_ylabel('psi y')
ax[1].set_xlabel('s [m]')

plt.show()
