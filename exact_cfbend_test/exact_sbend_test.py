#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
#import transformation_utilities as pycoord

from scipy.constants import c, eV, m_p
mp = 1.0e-9 * m_p * c**2/eV

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

mag_length = 4.0
mag_angle = np.pi/36 # 5 degrees

################

N_part = 16

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
sim.diagnostics = True
sim.slice_step_diagnostics = True


# domain decomposition & space charge mesh
sim.init_grids()

gamma = (mp+0.8)/mp

energy_MeV = (gamma - 1)*mp*1000.0
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

# particle 1 has offset pt
dpt[1] = -1.0e-3

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

cfb_ds = mag_length
mag_R = mag_length/mag_angle
cfb_knormal = np.array([1/mag_R, 0.0])
cfb_kskew = np.array([0.0, 0.0])

monitor = elements.BeamMonitor("monitor", backend="h5")
sim.lattice.extend(
    [
        monitor,
        # The bend
        #elements.ExactCFbend(ds=mag_length, k_normal=cfb_knormal, k_skew=cfb_kskew, nslice=16, name='foo'),
        elements.ExactSbend(ds=mag_length, phi=mag_angle*180.0/np.pi, nslice=1, name='bar'),
        monitor,
    ]
)

sim.track_particles()

# clean shutdown
sim.finalize()
