import sys, os
import numpy as np
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

def compare_rel(s, x, y, tol):
    print(f'{s}: {x} should be {y}, ', end='')
    if abs(x) < tol or abs(y) < tol:
        print(f'abs difference: {abs(x-y)}')
        assert x == approx(y, abs=tol)
    else:
        print(f'rel difference: {abs(x-y)/(abs(y))}')
        assert x == approx(y, rel=tol)

# open the data file
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iterations = list(series.iterations)


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

print('particle 0')
print('start')
print(x0[0], y0[0], t0[0], px0[0], py0[0], pt0[0])
print('finish')
print(x1[0], y1[0], t1[0], px1[0], py1[0], pt1[0])


# The 0 particle should stay at 0
assert abs(x1[0]) < 1.0e-14
assert abs(y1[0]) < 1.0e-14
assert abs(t1[0]) < 1.0e-14
assert abs(px1[0]) < 1.0e-14
assert abs(py1[0]) < 1.0e-14
assert abs(pt1[0]) < 1.0e-14

# The magnet is a 90 bend with a radius of curvature of 1 meter. Therefore
# orbit length through the element is pi/2.
R = 1.0

# particle 1
print('particle 1 coords')
print('start')
print(x0[1], y0[1], t0[1], px0[1], py0[1], pt0[1])
print('finish')
print(x1[1], y1[1], t1[1], px1[1], py1[1], pt1[1])
      
xoffs = -0.001
# shifted particle executes a circle or radius R starting the offset location
# leaving the magnet when it raches the 90 degree edge of the central trajectory.
xexit = np.sqrt(R**2 - xoffs**2)
xdiff = xexit-R
print('xdiff: ', xdiff)
print('rel diff x: ', abs(xdiff/x1[1])-1)
assert abs(xdiff/x1[1] - 1) < 1.0e-9

# test exit angle
# sin exit angle = xoffs/R which is also px
print('rel diff -theta/px1: ', abs((xoffs/R)/px1[1])-1)
assert abs((-xoffs/R)/px1[1] - 1) < 1.0e-12

# test transit time. The total angle is 90 degrees plus
# the angle arcsin(xoffs/R)
theta = np.pi/2 + np.arcsin(xoffs/R)
L0 = np.pi/2
L1 = R*theta
dt = (L1- L0)/beta_ref # same momentum/velocity
print('dt: ', dt, ' t1[1]: ', t1[1])
#assert abs(dt/t1[1] - 1) < 1.0e-13

# test particle 2 off momentum
print('particle 2')
print(x1[2], y1[2], t1[2], px1[2], py1[2], pt1[2])

ptoffs = pt0[2]
gamma = -beta_gamma_ref * pt1[2] + gamma_ref
betagamma = np.sqrt(gamma**2 - 1)
beta = betagamma/gamma

# test particle 3 xoffset oppoite to particle 1
print('particle 3 coords')
print('start')
print(x0[3], y0[3], t0[3], px0[3], py0[3], pt0[3])
print('finish')
print(x1[3], y1[3], t1[3], px1[3], py1[3], pt1[3])
      
xoffs = 0.001
# shifted particle executes a circle or radius R starting the offset location
# leaving the magnet when it raches the 90 degree edge of the central trajectory.
xexit = np.sqrt(R**2 - xoffs**2)
xdiff = xexit-R
print('xdiff: ', xdiff)
print('rel diff x: ', abs(xdiff/x1[3])-1)
assert abs(xdiff/x1[3] - 1) < 1.0e-9

# The radius of the off-momentum particle differes from the original radius by
# the ratio of momenta
Rnew = R * betagamma/beta_gamma_ref
# The center of circular motion is offset by the difference in radii

# particle intercepts exit face
xint = np.sqrt(Rnew**2 - (Rnew-R)**2) - R
print('xint: ', xint, ', x1[2]: ', x1[2])
assert abs(xint/x1[2] - 1) < 1.0e-13
# transit time
theta = np.arccos((Rnew-R)/Rnew)
L2 = Rnew * theta
dt = L2/beta - L0/beta_ref
print('dt: ', dt, ', t2[2]: ', t1[2])
print(abs(dt/t1[2] - 1))
assert abs(dt/t1[2] - 1) < 2.0e-13


