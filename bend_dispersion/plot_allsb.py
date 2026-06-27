#!//usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import pandas as pd

series_xcfb = io.Series("diags_xcfb/openPMD/monitor.h5", io.Access.read_only)
series_xsb = io.Series("diags_xsb/openPMD/monitor.h5", io.Access.read_only)
series_sb = io.Series("diags_sb/openPMD/monitor.h5", io.Access.read_only)


iters_xcfb = list(series_xcfb.iterations)
iters_xsb = list(series_xsb.iterations)
iters_sb = list(series_sb.iterations)


iter_xcfb0 = iters_xcfb[0]
iter_xcfb1 = iters_xcfb[-1]
iter_xsb0 = iters_xsb[0]
iter_xsb1 = iters_xsb[-1]
iter_sb0 = iters_sb[0]
iter_sb1 = iters_sb[-1]

beam_xcfb0 = series_xcfb.iterations[iter_xcfb0].particles["beam"]
beam_xcfb1 = series_xcfb.iterations[iter_xcfb1].particles["beam"]
beam_xsb0 = series_xsb.iterations[iter_xsb0].particles["beam"]
beam_xsb1 = series_xsb.iterations[iter_xsb1].particles["beam"]
beam_sb0 = series_sb.iterations[iter_sb0].particles["beam"]
beam_sb1 = series_sb.iterations[iter_sb1].particles["beam"]

df0 = beam_sb0.to_df()
df_xcfb1 = beam_xcfb1.to_df()
df_xsb1 = beam_xsb1.to_df()
df_sb1 = beam_sb1.to_df()

# Assume the initial state on all three simulations is the same

betagamma0 = beam_sb0.get_attribute('beta_gamma_ref')
gamma0 = beam_sb0.get_attribute('gamma_ref')
beta0 = beam_sb0.get_attribute('beta_ref')

# Calculate what the results should be

initial_pts = df0.loc[:, 'momentum_t']
initial_gammas = gamma0 - betagamma0*initial_pts
initial_betagammas = np.sqrt(initial_gammas**2 - 1)
initial_betas = initial_betagammas/initial_gammas

#print('initial_betagammas', initial_betagammas)

# magnet is radius 1m, angle pi/2.
R0 = 1
# bend radius for off-momentum particle is initial radius * new-momentum/original-momentum
Rs = (initial_betagammas/betagamma0) * R0
print('Rs: ', Rs)
# offset of the origin of the circular trajectory
dRs = Rs - R0
print('dRs: ', dRs)

# calculate final x position based on the offset of the orbit radius
rot_angles = np.arccos(dRs/Rs)
print('rot_angles: ', rot_angles)
xposs = Rs*np.sin(rot_angles) - R0
cT0 = np.pi/2 * R0/beta0
cTs = rot_angles * Rs/initial_betas
dTs = cTs-cT0

plt.figure()
plt.title('Final T')
plt.plot(df0['momentum_t'], df_xsb1['position_t'], 's', ms=10, label='exact Sbend')
plt.plot(df0['momentum_t'], df_xcfb1['position_t'], 'o', label='exact CFbend')
plt.plot(df0['momentum_t'], df_sb1['position_t'], 'd', label='sbend')
plt.plot(df0['momentum_t'], dTs, '-', label='calculated T')
plt.legend(loc='best')
plt.xlabel('dT')
plt.ylabel('T [m]')

plt.figure()
plt.title('Final x position')
plt.plot(df0['momentum_t'], df_xsb1['position_x'], 's', ms=10, label='exact Sbend')
plt.plot(df0['momentum_t'], df_xcfb1['position_x'], 'o', label='exact CFbend')
plt.plot(df0['momentum_t'], df_sb1['position_x'], 'd', label='sbend')
plt.plot(df0['momentum_t'], xposs, '-', label='calculated x position')
plt.legend(loc='best')
plt.xlabel('dT')
plt.ylabel('x position [m]')

plt.show()
sys.exit()

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

      
