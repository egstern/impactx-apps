#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Eric G. Stern, Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
#import transformation_utilities as pycoord

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

# Create a particle distribution with particles running from -x0 to +x0
# to figure out how they get read back in

x0 = 0.005

################

N_part = 2048

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = True
sim.slice_step_diagnostics = True

# save diags to specific directory
sim.diag_file_prefix = "diags_part"

# domain decomposition & space charge mesh
sim.init_grids()

charge_qe = 0.5e10  # used with space charge

#   reference particle
ref = sim.beam.ref
# set beam to IOTA protons at 2.5 MeV
ref.set_species("proton")
mp = ref.mass_MeV
#KE = 2.5
KE = 800.0
ref.set_kin_energy_MeV(KE)
ref.set_charge_qe(1.0)
qm_eev = 1.0 / (mp*1.0e6)  # 1/protom mass in eV
ref.z = 0

pc = sim.particle_container()

dx = np.zeros(N_part, dtype='d')
dpx = np.zeros(N_part, dtype='d')
dy = np.zeros(N_part, dtype='d')
dpy = np.zeros(N_part, dtype='d')
dt = np.zeros(N_part, dtype='d')
dpt = np.zeros(N_part, dtype='d')

# Set up the particle coordinates
dx = 2 * x0 * (np.arange(N_part) - N_part/2)/N_part
dy = 2 * x0 * (np.arange(N_part) - N_part/2)/N_part
dt = 2 * x0 * (np.arange(N_part) - N_part/2)/N_part


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
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge=charge_qe*ref.charge)

monitor = elements.BeamMonitor("monitor", backend="h5")

sim.lattice.extend(
    [
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()
