#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
from scipy.constants import eV, c

#import transformation_utilities as pycoord

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

################

N_part = 8

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True
#sim.slice_step_diagnostics = False

# domain decomposition & space charge mesh
sim.init_grids()

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_species("proton")
ref.set_kin_energy_MeV(800.0)
mp_mev = ref.mass_MeV

qm_eev = 1.0 / (mp_mev*1.0e-6)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()

dx = np.zeros(N_part, dtype='d')
dpx = np.zeros(N_part, dtype='d')
dy = np.zeros(N_part, dtype='d')
dpy = np.zeros(N_part, dtype='d')
dt = np.zeros(N_part, dtype='d')
dpt = np.zeros(N_part, dtype='d')

# all components are 0

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

bunch_charge_C = eV*6.7e12/81

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)


monitor = elements.BeamMonitor("monitor", backend="h5")
sim.lattice.clear()
#for i in range(1):
for i in range(400):
#    sim.lattice.append(elements.Drift(name=f"drift_{i:04}", ds=1.0, nslice=1))
# add segments 1/10 + 2/10 + 5/10 + 2/20
    sim.lattice.append(elements.Drift(name=f"drift_{i:04}a", ds=0.1, nslice=1))
    sim.lattice.append(elements.Drift(name=f"drift_{i:04}b", ds=0.2, nslice=1))
    sim.lattice.append(elements.Drift(name=f"drift_{i:04}c", ds=0.5, nslice=1))
    sim.lattice.append(elements.Drift(name=f"drift_{i:04}d", ds=0.2, nslice=1))

sim.lattice.append(monitor)

#sim.periods=15000
sim.verbose = 0
sim.periods=1000
sim.track_particles()

# clean shutdown
sim.finalize()
