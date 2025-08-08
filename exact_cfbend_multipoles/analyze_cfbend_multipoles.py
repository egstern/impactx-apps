import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

# open the data file
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iterations = list(series.iterations)

print('series.iterations: ', series.iterations)
for it in list(series.iterations):
    print(it)

start = series.iterations[iterations[0]]
finish = series.iterations[iterations[-1]]

start_beam = start.particles["beam"].to_df()
finish_beam = finish.particles["beam"].to_df()

mp = finish.particles["beam"].get_attribute('mass_ref')
beta_ref = finish.particles["beam"].get_attribute('beta_ref')
gamma_ref = finish.particles["beam"].get_attribute('gamma_ref')
beta_gamma_ref = finish.particles["beam"].get_attribute('beta_gamma_ref')


x0 = start_beam['position_x']
y0 = start_beam['position_y']
t0 = start_beam['position_t']
px0 = start_beam['momentum_x']
py0 = start_beam['momentum_y']
pt0 = start_beam['momentum_t']

# The 0 particle should remain at 0
x1 = finish_beam['position_x']
y1 = finish_beam['position_y']
t1 = finish_beam['position_t']
px1 = finish_beam['momentum_x']
py1 = finish_beam['momentum_y']
pt1 = finish_beam['momentum_t']

print('x0: ', x0)
print()
print('px0: ', px0)
print()
print('x1: ', x1)
print()
print('px1: ', px1)
print()

cfb_knormal = np.load('cfb_knormal.npy')[()]
print('cfb_knormal: ', cfb_knormal)

xx = np.arange(1024)*0.0016/1024
yy = (-cfb_knormal[1]*xx - 0.5*cfb_knormal[2]*xx**2)*0.001

plt.plot(x0, px1, '*')
plt.plot(xx, yy)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
