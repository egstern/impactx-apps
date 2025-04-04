#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
#import transformation_utilities as pycoord

import synergia
PCONST = synergia.foundation.pconstants
mp = PCONST.mp

import amrex.space3d as amr
from impactx import Config, ImpactX, elements



################

N_part = 16

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

refpart = synergia.foundation.Reference_particle(1, mp, 0.8+mp)
gamma = refpart.get_gamma()

energy_MeV = (refpart.get_total_energy() - mp)*1000.0
bunch_charge_C = 0.5e10  # used with space charge

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(1.0).set_mass_MeV(mp*1000).set_kin_energy_MeV(energy_MeV)
qm_eev = 1.0 / (mp*1.0e9)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()


dx = np.zeros(6, dtype='d')
dpx = np.zeros(6, dtype='d')
dy = np.zeros(6, dtype='d')
dpy = np.zeros(6, dtype='d')
dt = np.zeros(6, dtype='d')
dpt = np.zeros(6, dtype='d')

# particle 0 is at 0
# particle 1 has px offset and goes from -1mm to +1mm over 1 m
dx[1] = -0.001
dpx[1] = 0.002/np.sqrt(1 + 0.002**2)

# particle 2 travels straight but at pt = 1.0e-2
dpt[2] = 1.0e-2

# particle 3 travels from -0.001 to 0.001 at pt=1.0e-2
dy[3] = -0.001
dpy[3] = 0.002/np.sqrt(1 + 0.002**2)
dpt[3] = 1.0e-2

# particle 4 travels straight at pt=-1.0e-2
dpt[4] = -1.0e-2

# particle 5 travels from  0.001 to -0.001 at pt=-1.0e-2
dx[5] = 0.001
dpx[5] = -0.002/np.sqrt(1 + 0.002**2)
dpt[5] = -1.0e-2


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
        elements.ExactDrift(name="drift", ds=1.0, nslice=1),
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()
