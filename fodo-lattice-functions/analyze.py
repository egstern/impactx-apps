#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import pandas as pd

d = sys.argv[1]
print('opening files in directory: ', d)

rbc = pd.read_csv(f'{d}/reduced_beam_characteristics.0.0', delimiter='\\s+')
#print(rbc)

mx_s, mx_betx, mx_bety, mx_alfx, mx_alfy = np.loadtxt('fodo_lf.out', skiprows=52, usecols=(2,3,4,5,6), unpack=True)

# plt.figure()
# plt.title('MAD-X beta')
# plt.plot(mx_s, mx_betx, label='mx betx')
# plt.plot(mx_s, mx_bety, label='mx bety')
# plt.legend(loc='best')
# plt.xlabel('s')
# plt.ylabel('beta')

emitx = rbc['emittance_x']
emity = rbc['emittance_y']
sig_x = rbc['sig_x']
sig_y = rbc['sig_y']
beta_x = rbc['beta_x']
beta_y = rbc['beta_y']

s = rbc['s']

plt.figure()
plt.title('RBC emit x/y')
plt.plot(s, emitx, label='emit_x')
plt.plot(s, emity, label='emit_y')
plt.legend(loc='best')
plt.xlabel('step')
plt.ylabel('emittance')

plt.figure()
plt.title('RBC std x/y')
plt.plot(s, sig_x, label='std x')
plt.plot(s, sig_y, label='std y')
plt.legend(loc='best')
plt.xlabel('step')
plt.ylabel('std')

# plt.figure()
# plt.title('RBC beta x/y')
# plt.plot(s, beta_x, label='beta x')
# plt.plot(s, beta_y, label='beta y')
# plt.legend(loc='best')
# plt.xlabel('s')
# plt.ylabel('beta')

plt.figure()
plt.title('beta x/y')
plt.plot(mx_s, mx_betx, label='MAD-X betx')
plt.plot(mx_s, mx_bety, label='MAD-X  bety')
plt.plot(s, beta_x, label='RBC beta x')
plt.plot(s, beta_y, label='RBC beta y')
plt.legend(loc='best')
plt.xlabel('s')
plt.ylabel('Beta x/y')

plt.show()
