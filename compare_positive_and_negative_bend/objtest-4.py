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

################

def init_particles():
    # lp (local particles)
    lp = np.ndarray((32, 6))

    s2o2 = np.sqrt(2.0)/2.0
    s3o2 = np.sqrt(3.0)/2.0

    # 16 particles at radius of 1.0e-3 at 30degrees and 45 degrees offsets
    # around axis
    offset = 1.0e-3
    lp[0, 0] = offset
    lp[0, 2] = 0.0

    lp[1, 0] = offset*s3o2
    lp[1, 2] = offset*0.5

    lp[2, 0] = offset*s2o2
    lp[2, 2] = offset*s2o2

    lp[3, 0] = offset*0.5
    lp[3, 2] = offset*s3o2

    lp[4, 0] = 0.0
    lp[4, 2] = offset

    lp[5, 0] = -offset*0.5
    lp[5, 2] =  offset*s3o2

    lp[6, 0] = -offset*s2o2
    lp[6, 2] =  offset*s2o2

    lp[7, 0] = -offset*s3o2
    lp[7, 2] =  offset*0.5

    lp[8, 0] = -offset
    lp[8, 2] = 0.0

    lp[9, 0] = -offset*s3o2
    lp[9, 2] = -offset*0.5

    lp[10, 0] = -offset*s2o2
    lp[10, 2] = -offset*s2o2

    lp[11, 0] = -offset*0.5
    lp[11, 2] = -offset*s3o2

    lp[12, 0] = 0.0
    lp[12, 2] = -offset

    lp[13, 0] =  offset*0.5
    lp[13, 2] = -offset*s3o2

    lp[14, 0] =  offset*s2o2
    lp[14, 2] = -offset*s2o2

    lp[15, 0] =  offset*s3o2
    lp[15, 2] = -offset*0.5

    # repeat same pattern shifted 0.5mm down 0.25mm
    x2_offset = 0.0005
    y2_offset = -0.00025
    for i in range(16,32):
        lp[i, 0] = lp[i-16, 0] + x2_offset
        lp[i, 2] = lp[i-16, 2] + y2_offset

    return lp


################


def create_sim():
    if not hasattr(create_sim, 'cnt'):
        create_sim.cnt = 1
    else:
        create_sim.cnt = create_sim.cnt + 1

    print(f'create_sim, iteration {create_sim.cnt}')

    sim = ImpactX()

    # set numerical parameters and IO control
    sim.particle_shape = 2  # B-spline order
    sim.space_charge = False
    # sim.diagnostics = False  # benchmarking
    if create_sim.cnt == 1:
        sim.diagnostics = True
        print(f'first time, what is diagnostics: {sim.diagnostics}')
    if create_sim.cnt > 1:
        # 2nd time turn off diagnostics
        print('turning off diagnostics')
        sim.diagnostics = False
    sim.slice_step_diagnostics = False

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


    dx = np.zeros(N_part, dtype='d')
    dpx = np.zeros(N_part, dtype='d')
    dy = np.zeros(N_part, dtype='d')
    dpy = np.zeros(N_part, dtype='d')
    dt = np.zeros(N_part, dtype='d')
    dpt = np.zeros(N_part, dtype='d')


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


    particles = init_particles()

    for p_dx in particles[:, 0]:
        dx_podv.push_back(p_dx)
    for p_dy in particles[:, 2]:
        dy_podv.push_back(p_dy)
    for p_dt in particles[:, 4]:
        dt_podv.push_back(p_dt)
    for p_dpx in particles[:, 1]:
        dpx_podv.push_back(p_dpx)
    for p_dpy in particles[:, 3]:
        dpy_podv.push_back(p_dpy)
    for p_dpt in particles[:, 5]:
        dpt_podv.push_back(p_dpt)

    pc.add_n_particles(
        dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
    )
    return sim


def set_lattice(sim, bendangle):

    if bendangle > 0:
        nm = "monitor_pos"
    else:
        nm = "monitor_neg"

    monitor = elements.BeamMonitor(nm, backend="h5")
    sim.lattice.extend(
        [
            monitor,
            # Sbend, 90 degree bend radius 1 m
            elements.ExactSbend(np.pi/2, bendangle, name="sbend", nslice=4),
            monitor,
        ]
    )
    return

def do_sim(bendangle):

    sim = create_sim()

    set_lattice(sim, bendangle)

    sim.track_particles()

    # clean shutdown
    sim.finalize()
    return

################

def main():
    do_sim(-90.0)
    do_sim(+90.0)
    return

if __name__ == "__main__":
    main()
    pass
