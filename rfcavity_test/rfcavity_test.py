#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
#import transformation_utilities as pycoord

import synergia
PCONST = synergia.foundation.pconstants
mp = PCONST.mp
c = PCONST.c

import amrex.space3d as amr
from impactx import Config, ImpactX, elements



################

N_part = 1024

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = False

# domain decomposition & space charge mesh
sim.init_grids()

refpart = synergia.foundation.Reference_particle(1, mp, 0.8+mp)
gamma = refpart.get_gamma()
beta = refpart.get_beta()

energy_MeV = (refpart.get_total_energy() - mp)*1000.0
bunch_charge_C = 0.5e10  # used with space charge

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(1.0).set_mass_MeV(mp*1000).set_kin_energy_MeV(energy_MeV)
qm_eev = 1.0 / (mp*1.0e9)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()

# We'rd going use parameters like Booster cavities.
# L = 2.35. KE = 0.8 GeV, 22 cavities, 1 MV total voltage, ring
# length = 484.202752, harmonic number=84
print('beta: ', beta)
V = 1.0e-3/22 # MV
h = 84 # harmonic number 
L = 474.202752 # m
freq = h * beta * c/L
bucket_length = L/h

print('frequency: ', freq)
print('bucket length: ', bucket_length)

dx = np.zeros(N_part, dtype='d')
dpx = np.zeros(N_part, dtype='d')
dy = np.zeros(N_part, dtype='d')
dpy = np.zeros(N_part, dtype='d')
dt = np.zeros(N_part, dtype='d')
dpt = np.zeros(N_part, dtype='d')

# distribute the particles at uniform phase from -pi to +511/512 pi in the
# bucket. particle 512 should be at 0 phase.

dl = bucket_length/N_part * (1/beta)
for i in range(N_part):
    dt[i] = dl * (i-N_part//2)

print('most negative phase particle (0): ', dt[0])
print(f'middle phase particle ({N_part//2}): ', dt[N_part//2])
print(f'most positive phase particle ({N_part-1}): ', dt[-1])

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
sim.lattice.extend(
    [
        monitor,
        # below transition phase=-90.0
        elements.ShortRF((V/mp), freq, phase=-90.0, name="rfc"),
        # above transition phase should be 90
        # elements.ShortRF((V/mp), freq, phase=90.0, name="rfc"),
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()
