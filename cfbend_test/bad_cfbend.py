#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import scipy.constants as constants
import openpmd_api as io
import pandas as pd

mp = 1.0e-9 * constants.m_p * constants.c**2/constants.elementary_charge
print('mass proton: ', mp, 'GeV')

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

################

N_part = 16

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
sim.diagnostics = True  # benchmarking
sim.step_diagnostics = True
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# Set proton kinetic energy 0.8 GeV
E = 0.8 + mp
g = E/mp # energy
bg = np.sqrt(g**2 - 1) # beta*gamma

energy_MeV = (E - mp)*1000.0
bunch_charge_C = 0.5e10  # used with space charge

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(1.0).set_mass_MeV(mp*1000).set_kin_energy_MeV(energy_MeV)
qm_eev = 1.0 / (mp*1.0e9)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()

dx = np.zeros(N_part, dtype='d')
dpx = np.zeros(N_part, dtype='d')
dy = np.zeros(N_part, dtype='d')
dpy = np.zeros(N_part, dtype='d')
dt = np.zeros(N_part, dtype='d')
dpt = np.zeros(N_part, dtype='d')

# particle 0 is at 0

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
        # CFbend, 15 degree bend radius 1 m, length is pi/12, k=0 # no
        # quad field
        #elements.CFbend(ds=np.pi/12, rc=1.0, k=1.0e-180, name="cfend", nslice=25),
        elements.CFbend(ds=np.pi/12, rc=1.0, k=0.0, name="cfend", nslice=25),
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()

# open the data file
series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iterations = list(series.iterations)

print('iterations in file:', iterations)

initial = series.iterations[iterations[0]]
final = series.iterations[iterations[-1]]

initial_beam = initial.particles["beam"].to_df()
final_beam = final.particles["beam"].to_df()

initial_x = initial_beam['position_x']
initial_xp = initial_beam['momentum_x']
initial_y = initial_beam['position_y']
initial_yp = initial_beam['momentum_y']
initial_t = initial_beam['position_t']
initial_pt = initial_beam['momentum_t'] 

final_x = final_beam['position_x']
final_xp = final_beam['momentum_x']
final_y = final_beam['position_y']
final_yp = final_beam['momentum_y']
final_t = final_beam['position_t']
final_pt = final_beam['momentum_t']

print('initial particle 0 coords: ',
      initial_x[0], initial_xp[0],
      initial_y[0], initial_yp[0],
      initial_t[0], initial_pt[0])

print('final particle 0 coords: ',
      final_x[0], final_xp[0],
      final_y[0], final_yp[0],
      final_t[0], final_pt[0])

rbc = pd.read_csv('diags/reduced_beam_characteristics.0.0', delimiter=r'\s+')
print('reduced beam characteristics:')
print(rbc)
print()

ref_particle = pd.read_csv('diags/ref_particle.0.0', delimiter=r'\s+')
print('ref particle evolution')
print(ref_particle)

series.close()


      
