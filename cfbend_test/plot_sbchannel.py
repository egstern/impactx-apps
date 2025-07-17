#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.constants as constants

def norm(x):
    print('norm: x: ', x)
    sq = np.dot(x,x)
    return np.sqrt(sq)


# calculate angle for off-momentum trajectory
def off_momentum_angle(refp, dpp, refangle, reflen):
    R1 = reflen/refangle
    R2 = R1 * (1+dpp)
    D = R2-R1
    # on-momentum trajectory is a arc of radius R1 centered at
    # the origin starting at (x,y)=(0,R1) and exiting at the line
    # with angle to the y axis of refangle.

    # the off-momentum trajectory is an arc of radius R2 centered
    # at (x, y) = (0, -R2). It exits at the line going through the
    # progom  with angle to the y axis of refangle.

    #  x**2 + (y+R2-R1)**2 = R2**2; D = R2-R1
    #  x**2 + (y + D)**2 = R2**2
    #     and
    #  y = cot(refangle) * x
    #  x = tan(refangle) * y

    # tan**2 y**2 + y**2 + 2*D*y + D**2 = R2**2
    # sec**2 y**2 + 2*D*y + D**2 = R2**2
    sectheta = 1.0/np.cos(refangle)
    y = -D + np.sqrt(D**2 + (R2**2 - D**2) * sectheta**2 )/sectheta**2
    x = np.tan(refangle) * y
    print('x intersection: ', x)
    print('y intersection: ', y)
    # # on-momentum intersection
    # x0 = R1*np.sin(refangle)
    # y0 = R1*np.cos(refangle)
    # print('on-momentum intersection')
    # print('x: ', x0)
    # print('y: ', y0)
    # # what's the angle of the off-momentum intersection?
    # print('off-momentum angle: ', np.arctan2(x, y))

    # unit vector from off-momentum radius to off-momentum exit point
    r2 = np.array([x, y+D])
    print('r2: ', r2)
    r2 = r2/norm(r2)
    print('normalized r2: ', r2)
    # unit vector for the on-momentum radius
    r1 = np.array([np.sin(refangle), np.cos(refangle)])
    # take cross product to get sin angle
    print('r1: ', r1)
    print('r2: ', r2)
    r1xr2 = r1[0]*r2[1] - r1[1]*r2[0]
    print('r1xr2: ', r1xr2)
    return np.arcsin(r1xr2)


h5a = h5py.File('sbtracks.h5', 'r')
h5b = h5py.File('nbtracks.h5', 'r')

print('h5a.keys(): ', h5a.keys())
print('h5b.keys(): ', h5b.keys())

trksa = h5a.get('track_coords')[()]
trksb = h5b.get('track_coords')[()]


pz = h5a.get('pz')[()]
mp = h5a.get('mass')[()]
print('mass: ', mp)
print('pz0: ', pz)
print('dp/p: ', trksa[0, :11, 5])

h5a.close()
h5b.close()

# this magnet:
Flen = 3.0#;  ! focusing magnet length
Fk1 = 0.055# ! focusing magnet gradient
FR = 40.0#   ! focusing magnet radius of curvature
Fang = Flen/FR# ; ! angle

offmomange = off_momentum_angle(pz, 5.0e-3, Fang, Flen)
print('angle: ', offmomange)

sb_madx = np.loadtxt('sbchannel.txtone', skiprows=21, usecols=(2,3,4,5,6,7))
nb_madx = np.loadtxt('nbchannel.txtone', skiprows=21, usecols=(2,3,4,5,6,7))
print('loadtxt finalx: ', nb_madx[:, 0])
print('loadtxt pt: ', nb_madx[:, 5])

Brho = 1.0e9/constants.c
BL = Fang * Brho

initx = trksa[0, :11, 0]
dpop = trksa[0, 0, 5]
final_xa = trksa[-1, :11, 0]
final_xpa = trksa[-1, :11, 1]
final_xb = trksb[-1, :11, 0]
final_xpb = trksb[-1, :11, 1]

# plot the nominal focusing
focusing = -Fk1*Flen*initx + offmomange

plt.figure()
plt.title('x momentum')
plt.plot(initx, final_xpa, 'o', label='px vs. init position with k1')
plt.plot(initx, final_xpb, 'o', label='px vs. init position without k1')
plt.plot(initx, sb_madx[:, 1], label='madx px vs. init position with k1')
plt.plot(initx, nb_madx[:, 1], label='madx px vs. init position without k1')
plt.plot(initx, focusing, '-', label='k1 focussing')

plt.legend(loc='best')
plt.xlabel('init x')
plt.ylabel('px')

plt.figure()
plt.title('x offiset')
plt.plot(initx, final_xa, 'o', label='final x synergia with k1')
plt.plot(initx, final_xb, 'o', label='final x synergia without k1')
plt.plot(initx, sb_madx[:, 0], '*', label='final x madx with k1')
plt.plot(initx, nb_madx[:, 0], '*', label='final x madx without k1')
plt.legend(loc='best')
plt.xlabel('init x')
plt.ylabel('x')

plt.show()
