#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np

from scipy.constants import c, e as qe, m_p
mp_gev = m_p * c**2 * 1.0e-9/qe

import amrex.space3d as amr
from impactx import Config, ImpactX, elements


# beam is protons with momentum of 1.5 GeV/c (from Synergia2 test)
pbeam = 1.5
Ebeam = np.sqrt(mp_gev** + pbeam**2)
betagamma0 = pbeam/mp_gev
gamma0 = np.sqrt(betagamma0**1 + 1)
beta0 = betagamma0/gamma0

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


def init_and_run_sim(lattice):
    sim = ImpactX()
    # set numerical parameters and IO control
    sim.particle_shape = 2  # B-spline order
    sim.space_charge = False
    # sim.diagnostics = False  # benchmarking
    sim.slice_step_diagnostics = False

    # domain decomposition & space charge mesh
    sim.init_grids()

    energy_MeV = mp_gev * 1000
    bunch_charge_C = 0.5e10

    #   reference particle
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe(1.0).set_mass_MeV(mp_gev*1000).set_kin_energy_MeV(energy_MeV)
    qm_eev = 1.0 / (mp_gev*1.0e9)  # 1/protom mass  in eV
    ref.z = 0

    pc = sim.particle_container()

    particles = init_particles()

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

    sim.lattice.extend(lattice)

    sim.track_particles()

    sim.finalize()

#**********************************************************************

def run_k1():
    print("enter run_k1")

    nm = "k1"

    # create the beamline
    monitor = elements.BeamMonitor(f'monitor_{nm}.h5', backend="h5")
    # mp_str = 0.1;
    # mpole: multipole, knl={0.0, mp_str};
    mp_str = 0.1;
    elem = elements.Multipole(2, mp_str, 0.0, name=nm)

    lattice = [monitor, elem, monitor]

    init_and_run_sim(lattice)
    
    print('exit run_k1')
    pass

#**********************************************************************

def run_k1s():
    print("enter run_k1s")

    nm = "k1s"

    # create the beamline
    monitor = elements.BeamMonitor(f'monitor_{nm}.h5', backend="h5")
    # mp_str = 0.1;
    # mpole: multipole, ksl={0.0, mp_str};
    mp_str = 0.1;
    elem = elements.Multipole(2, 0.0, mp_str, name=nm)

    lattice = [monitor, elem, monitor]

    init_and_run_sim(lattice)
    print("run_k1s: after sim = init_sim")

    print("exit run_k1s")
    pass

#**********************************************************************


def main():
    print("before run_k1")
    run_k1()
    print("after run_k1 before run_k1s")
    run_k1s()
    print("after run_k1s")

    return

if __name__ == "__main__":
    main()
    pass
