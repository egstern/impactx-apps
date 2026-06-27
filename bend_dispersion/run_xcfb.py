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

# This test runs particles at different momenta through a 1m radius
# dipole bending 90 degrees.

R0 = 1.0
angle = np.pi/2 # 90 degrees
length = R0 * angle

################

N_part = 16

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = True
sim.slice_step_diagnostics = True

# save diags to specific cfb directory
sim.diag_file_prefix = "diags_xcfb"

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

# arrange 16 particles with dp/p ranging from [-0.01, 0.01)
dpop_max = 0.01
dpops = 2.0 * dpop_max * np.arange(-N_part/2, N_part/2)/N_part
print('dpops: ', dpops)
# Leave all particle coordinates 0 except for dpt which has to
# be calculated from dp/p

bg0 = ref.beta_gamma # beta*gamam
g0 = ref.gamma # gamma

beta_gammas = (1+dpops) * bg0
print('beta_gammas: ', beta_gammas)
gammas = np.sqrt(beta_gammas**2 + 1 )
print('gammas: ', gammas)
dpt = (g0 - gammas)/bg0
print("dpt: ", dpt)

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

cfb_ds = length
mag_RC = R0
cfb_knormal = np.array([1/mag_RC, 0.0])
cfb_kskew = np.array([0.0, 0.0])

monitor = elements.BeamMonitor("monitor", backend="h5")

sim.lattice.extend(
    [
        monitor,
        elements.ExactCFbend(ds=cfb_ds, k_normal=cfb_knormal, k_skew=cfb_kskew, nslice=1, name='foo'),
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()
