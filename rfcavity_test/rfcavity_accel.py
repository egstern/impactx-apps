#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import sys, os
import cmath
import numpy as np
#import transformation_utilities as pycoord

from scipy.constants import m_p, c, eV

mp_mev = 1.0e-6 * m_p * c**2/eV


import amrex.space3d as amr
from impactx import Config, ImpactX, elements

################

N_part = 16

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.diagnostics = True
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

kin_energy_mev = 800.0

bunch_charge_C = 0.5e10  # used with space charge

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(1.0).set_mass_MeV(mp_mev).set_kin_energy_MeV(kin_energy_mev)
beta = ref.beta

qm_eev = 1.0 / (mp_mev*1.0e6)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()

# We're going use parameters like Booster cavities.
# L = 2.35. KE = 0.8 GeV, 22 cavities, 1 MV total voltage, ring
# length = 474.202752, harmonic number=84
print('beta: ', beta)
V = 1.0/22 # 1 MV spread over 22 cavities
#V = 0.2/22 # 200 KV spread over 22 cavities
h = 84 # harmonic number 
L = 474.202752 # m

freq = h * beta * c/L
bucket_length = L/h

print('frequency: ', freq)
print('bucket length: ', bucket_length, 'm')
print('cavity voltage [MV]: ', V)

dx = np.zeros(N_part, dtype='d')
dpx = np.zeros(N_part, dtype='d')
dy = np.zeros(N_part, dtype='d')
dpy = np.zeros(N_part, dtype='d')
dt = np.zeros(N_part, dtype='d')
dpt = np.zeros(N_part, dtype='d')



#particle 0: at z=0
#particle [1:5) at z=0 with px,py components
#particle [5:9) at z=+bucket_length/4 with px, py components
#particle [9:13) at z=-bucket_length/4 with px, py components

# distribute the particles at uniform phase from -pi to +511/512 pi in the
# bucket. particle 512 should be at 0 phase.


for i in range(1,5):
    dpx[i] =  1.0e-2 * ((1j)**i).real
    dpy[i] =  1.0e-2 * ((1j)**i).imag

for i in range(5,9):
    dpx[i] =  1.0e-2 * ((1j)**i).real
    dpy[i] =  1.0e-2 * ((1j)**i).imag
    dt[i] = bucket_length/(4*beta)

for i in range(9,13):
    dpx[i] =  1.0e-2 * ((1j)**i).real
    dpy[i] =  1.0e-2 * ((1j)**i).imag
    dt[i] = -bucket_length/(4*beta)

if not Config.have_gpu:  # initialize using cpu-based PODVectors
    dx_podv = amr.PODVector_real_std()
    dy_podv = amr.PODVector_real_std()
    dt_podv = amr.PODVector_real_std()
    dpx_podv = amr.PODVector_real_std()
    dpy_podv = amr.PODVector_real_std()
    dpt_podv = amr.PODVector_real_std()
else:  # initialize on device using arena/gpu-based PODVectors
    dx_podv = amr.PODVector_real_arena()
    dy_podv = amr.PODVector_real_arena()
    dt_podv = amr.PODVector_real_arena()
    dpx_podv = amr.PODVector_real_arena()
    dpy_podv = amr.PODVector_real_arena()
    dpt_podv = amr.PODVector_real_arena()

for p_dx in dx:
    dx_podv.push_back(p_dx)
for p_dy in dy:
    dy_podv.push_back(p_dy)
for p_dt in dt:
    dt_podv.push_back(p_dt)
for p_dpx in dpx:
    dpx_podv.push_back(p_dpx)
for p_dpy in dpy:
    dpy_podv.push_back(p_dpy)
for p_dpt in dpt:
    dpt_podv.push_back(p_dpt)

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)

monitor = elements.BeamMonitor("monitor", backend="h5")
# put in two caviites
sim.lattice.extend(
    [
        monitor,
        # below transition phase=-90.0
        elements.ShortRF((V/mp_mev), freq, phase=0.0, name="rfc"),
        monitor,
        elements.ExactDrift(1.0, name="drift1"),
        elements.ShortRF((V/mp_mev), freq, phase=0.0, name="rfc"),
        monitor,
    ]
)

print('Lattice: ')
for i in sim.lattice:
    print(i)

sim.track_particles()

# clean shutdown
sim.finalize()
