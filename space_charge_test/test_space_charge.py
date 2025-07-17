#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
import openpmd_api as io

from scipy.constants import c, e as qe, m_p
mp_gev = m_p * c**2 * 1.0e-9/qe

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

# set Lorentz parameters
# beta*gamma = 15
# gamma = 113
# betagamma0 = 15/112
# gamma0 = 113/112
# beta0 = 15/113
Ebeam = 0.4 + mp_gev
gamma0 = Ebeam/mp_gev
betagamma0 = np.sqrt(gamma0**2-1)
beta0 = betagamma0/gamma0

pbeam = mp_gev * betagamma0
#Ebeam = mp_gev * gamma0

len_channel = 1.0
# number of particles
# 1 + (odd number) * 24 to expand the domain on both sides
#npart = 1 + 32767*24

# even number of segments. particles live in the center of each segment
# symmetrically going positive and negative
npart = 1+32768*24

rod_charge = 0.5e11 # about a booster bunch

rod_radius = 1.0e-6  # radius of rod of charge
probe = 1.0e-3 # offset of probe particle
probe = 5.0e-6
#**********************************************************************

def init_particles():
    # lp (local particles)
    lp = np.ndarray((npart, 6))

    s15 = (np.sqrt(6) - np.sqrt(2))/4
    c15 = (np.sqrt(6) + np.sqrt(2))/4
    s30 = 0.5
    c30 = np.sqrt(3)/2
    c45 = np.sqrt(2)/2
    s45 = c45
    s60 = c30
    c60 = s30
    s75 = c15
    c75 = s15

    ns = int((npart - 1)/24)
    ds = len_channel/ns
    print('ns: ', ns)
    print('ds: ', ds)

    r = rod_radius
    lp[1:npart:24, 0] = r
    lp[1:npart:24, 2] = 0.0

    lp[2:npart:24, 0] = r*c15
    lp[2:npart:24, 2] = r*s15

    lp[3:npart:24, 0] = r*c30
    lp[3:npart:24, 2] = r*s30

    lp[4:npart:24, 0] = r*c45
    lp[4:npart:24, 2] = r*s45

    lp[5:npart:24, 0] = r*c60
    lp[5:npart:24, 2] = r*s60

    lp[6:npart:24, 0] = r*c75
    lp[6:npart:24, 2] = r*s75

    # reflect the first quadrant particles 90 degrees
    for j in range(1,7):
        lp[j+6:npart:24, 0] = -lp[j:npart:24, 2]
        lp[j+6:npart:24, 2] =  lp[j:npart:24, 0]

    # 3rd and fourth quadrant is sign-flipped first and second  quadrant
    for j in range(1,13):
        lp[j+12:npart:24, 0] = -lp[j:npart:24, 0]
        lp[j+12:npart:24, 2] = -lp[j:npart:24, 2]

    lp[0, 0] = probe

    # set longitudinal position
    s0 = -len_channel/2 + ds/2
    lpos = 1
    for j in range(1, ns+1):
        lp[lpos:lpos+24, 4] = s0/beta0
        s0 = s0 + ds
        lpos = lpos+24

    print('first part of lpos:')
    print(lp[:25, 4]*beta0)
    print('last part of lpos:')
    print(lp[-1-25:, 4]*beta0)
    return lp


#**********************************************************************



def create_sim():

    print("enter create_sim")
    sim = ImpactX()

    print("after ImpactX()")

    # set numerical parameters and IO control
    sim.particle_shape = 2  # B-spline order

    #sim.n_cell = [64, 64, 256]  # [72, 72, 64] for increased precision
    sim.n_cell = [40, 40, 40]  # [72, 72, 64] for increased precision
    sim.particle_shape = 2  # B-spline order
    sim.space_charge = "3D"
    sim.poisson_solver = "fft"
    sim.prob_relative = [1.1]

    # sim.diagnostics = False  # benchmarking
    #sim.diagnostics = False
    sim.slice_step_diagnostics = False

    # domain decomposition & space charge mesh
    sim.init_grids()

    energy_MeV = mp_gev * 1000
    bunch_charge_C = rod_charge

    #   reference particle
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe(1.0).set_mass_MeV(mp_gev*1000).set_kin_energy_MeV(0.4*1000)
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

    print('len(dx_podv): ', len(dx_podv))
    print(dir(pc))
    print('pc.total_numer_of_particles: ', pc.total_number_of_particles())

    return sim

#**********************************************************************

def create_lattice():
    monitor = elements.BeamMonitor('monitor.h5', backend="h5")

    drift_elem = elements.ExactDrift(name="drift", ds=len_channel, nslice=10)

    lattice = [monitor, drift_elem, monitor]

    return lattice

#**********************************************************************

def get_SC_kick():
    series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)
    iterations = list(series.iterations)

    start = series.iterations[iterations[0]]
    finish = series.iterations[iterations[-1]]

    start_beam = start.particles["beam"].to_df()
    finish_beam = finish.particles["beam"].to_df()

    kick = np.array([finish_beam['momentum_x'][0],
                     finish_beam['momentum_y'][0],
                     finish_beam['momentum_t'][0]])
    series.close()
    return kick

#**********************************************************************

def run_channel():

    sim = create_sim()

    sim.lattice.extend(create_lattice())

    print('lattice in sim')
    for e in sim.lattice:
        print(e)
    print()

    print(dir(sim.particle_container))

    sim.track_particles()

    sim.finalize()

    SCkick = get_SC_kick()

    print('SCkick: ', SCkick)

    return

#**********************************************************************

def main():
    run_channel()
    return

if __name__ == "__main__":
    main()
    
