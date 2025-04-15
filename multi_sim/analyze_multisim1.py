import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

slices = [1,2,4,8,16]

slicedf = {}

beta_ref = None
for s in slices:
    print('reading slices: ', s)
    series = io.Series(f"diags/openPMD/monitor_{s:02d}.h5", io.Access.read_only)
    lastiter = list(series.iterations)[-1]
    if not beta_ref:
        beta_ref = series.iterations[lastiter].particles['beam'].get_attribute('beta_ref')
        gamma_ref = series.iterations[lastiter].particles['beam'].get_attribute('gamma_ref')
        beta_gamma_ref = series.iterations[lastiter].particles['beam'].get_attribute('beta_gamma_ref')
    df = series.iterations[lastiter].particles['beam'].to_df();
    #print(df)
    #print()
    slicedf[s] = df
    series.close()

npart = slicedf[1].shape[0]
pos_x = np.ndarray((npart, 0))
mom_x = np.ndarray((npart, 0))
pos_y = np.ndarray((npart, 0))
pos_t = np.ndarray((npart, 0))
mom_t = np.ndarray((npart, 0))
print(pos_x.shape)

for s in slices:
    row = slicedf[ s]['position_x'].to_numpy().reshape(npart, 1)
    pos_x = np.hstack((pos_x, row))
    row = slicedf[s]['momentum_x'].to_numpy().reshape(npart, 1)
    mom_x = np.hstack((mom_x, row))
    row = slicedf[s]['position_y'].to_numpy().reshape(npart, 1)
    pos_y = np.hstack((pos_y, row))
    row = slicedf[s]['position_t'].to_numpy().reshape(npart, 1)
    pos_t = np.hstack((pos_t, row))
    row = slicedf[s]['momentum_t'].to_numpy().reshape(npart, 1)
    mom_t = np.hstack((mom_t, row))

plt.figure()
plt.title('0 particle coordinatesx position vs. slices (should all be 0')
plt.plot(slices, pos_x[0, :], label='position_x')
plt.plot(slices, mom_x[0, :], label='momentum_x')
plt.plot(slices, pos_t[0, :], label='position_t')
plt.plot(slices, pos_y[0, :], label='position_y')
plt.legend(loc='best')
ax = plt.gca()
ax.set_xscale('log', base=2)

plt.xlabel('slices')

# calculate what the correct path should be for x offset particle
f, ax = plt.subplots(3, 1, sharex=True)


# The magnet is a 90 bend with a radius of curvature of 1 meter. Therefore
# orbit length through the element is pi/2.
R = 1.0
xoffs = -0.001
xexit = np.sqrt(R**2 - xoffs**2)
xdiff = xexit-R

ax[0].plot(slices, pos_x[1, :]-xoffs, label='position_x')
ax[0].legend(loc='best')
ax[0].set_xscale('log', base=2)
# test exit angle
# sin exit angle = xoffs/R which is also px
ax[1].plot(slices, -xoffs/R - pos_x[1, :], label='momentum_x')
ax[1].legend(loc='best')
ax[1].set_xscale('log', base=2)

# test transit time. The total angle is 90 degrees plus
# the angle arcsin(xoffs/R)
theta = np.pi/2 + np.arcsin(xoffs/R)
L0 = np.pi/2
L1 = R*theta
dt = (L1- L0)/beta_ref # same momentum/velocity
ax[2].plot(slices, pos_t[1, :]-dt, label='position_t')
ax[2].legend(loc='best')
ax[2].set_xscale('log', base=2)

xoffs_df = pd.DataFrame(index=slices, columns=['pos-x', 'mom-x', 'pos-t'],
                        data=np.vstack(((xdiff - pos_x[1,:]),
                                        (-xoffs/R - pos_x[1, :]),
                                        (dt-pos_t[1, :]) ) ).transpose() )


#pd.set_option('display.precision', 10)

xrel_df = pd.DataFrame(index=slices,
                       columns=['pos-x', 'mom-x', 'pos-t'],
                       data=np.vstack((abs(pos_x[1,:]/xdiff-1),
                                        abs(mom_x[1,:]/(-xoffs/R) - 1),
                                        abs(pos_t[1, :]/dt - 1 ) )).transpose() )
                       

print(xoffs_df)
print()
print(xrel_df)

# test particle 2 off momentum
ptoffs = mom_t[2, 0]
gamma = -beta_gamma_ref * ptoffs + gamma_ref
betagamma = np.sqrt(gamma**2 - 1)
beta = betagamma/gamma


# The radius of the off-momentum particle differes from the original radius by
# the ratio of momenta
Rnew = R * betagamma/beta_gamma_ref
#print('R: ', R, ', Rnew: ', Rnew)

# The center of circular motion is offset by the difference in radii

# particle intercepts exit face
xint = np.sqrt(Rnew**2 - (Rnew-R)**2) - R
#transit time
theta = np.arccos((Rnew-R)/Rnew)
L2 = Rnew * theta
dt = L2/beta - L0/beta_ref

xrel2_df =pd.DataFrame(index=slices,
                       columns=['pos-x', 'pos-t'],
                       data=np.vstack((abs(pos_x[2,:]/xint-1),
                                        abs(pos_t[2, :]/dt - 1 ) )).transpose() )
                       
print(xrel2_df)

#ax[0]
#plt.title(particle 1 x position
plt.show()

