#!//usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import pandas as pd

series = io.Series("diags_xsb/openPMD/monitor.h5", io.Access.read_only)

iters = list(series.iterations)

iter0 = iters[0]
iter1 = iters[-1]

beam0 = series.iterations[iter0].particles["beam"]
beam1 = series.iterations[iter1].particles["beam"]

df0 = beam0.to_df()
df1 = beam1.to_df()

betagamma0 = beam0.get_attribute('beta_gamma_ref')
gamma0 = beam0.get_attribute('gamma_ref')
beta0 = beam0.get_attribute('beta_ref')

print('Initial T')
print(df0.loc[:, 'position_t'])
print()
print('Initial PT')
print(df0.loc[:, 'momentum_t'])
print()
print('Final T')
print(df1.loc[:, 'position_t'])
print()
print('Final PT')
print(df1.loc[:, 'momentum_t'])
print()
print('Final x')
print(df1.loc[:, 'position_x'])
print()

series.close()

      
