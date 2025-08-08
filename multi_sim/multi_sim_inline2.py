#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Ryan Sandberg, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
#import transformation_utilities as pycoord
from scipy.constants import c, e as qe, m_p
mp_gev = m_p * c**2 * 1.0e-9/qe

import amrex.space3d as amr
from impactx import Config, ImpactX, elements

npart = 16

# beam is protons with kinetic energy of 0.8 GeV
Ebeam = mp_gev + 0.8
gamma0 = Ebeam/mp_gev
betagamma0 = np.sqrt(gamma0**2 - 1)
beta0 = betagamma0/gamma0

# Create the simulator for n slices
def create_sim(nslice):
    return sim

def main():
    # slices to try
    slices = [1,2,4,8,16]
    #slices = [1, 5]
    #slices = [5]

    sims = {}
    for nslice in slices:

        sim = ImpactX()

        # set numerical parameters and IO control
        sim.particle_shape = 2  # B-spline order
        sim.space_charge = False
        # sim.diagnostics = False  # benchmarking
        sim.slice_step_diagnostics = False

        # domain decomposition & space charge mesh
        sim.init_grids()

        energy_MeV = (Ebeam - mp_gev)*1000.0
        bunch_charge_C = 0.5e10  # used with space charge

        #   reference particle
        ref = sim.particle_container().ref_particle()
        ref.set_charge_qe(1.0).set_mass_MeV(mp_gev*1000).set_kin_energy_MeV(energy_MeV)
        qm_eev = 1.0 / (mp_gev*1.0e9)  # 1/proton mass  in eV
        ref.z = 0

        pc = sim.particle_container()

        dx = np.zeros(npart, dtype='d')
        dpx = np.zeros(npart, dtype='d')
        dy = np.zeros(npart, dtype='d')
        dpy = np.zeros(npart, dtype='d')
        dt = np.zeros(npart, dtype='d')
        dpt = np.zeros(npart, dtype='d')

        # particle 0 is at 0

        # particle 1 has px offset and goes from -1mm to +1mm over 1 m
        xoffs = -0.001
        dx[1] = xoffs

        # particle 2 has a pt offset of 0.01
        dpt[2] = 0.01

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

        # Create the beamline
        monitor = elements.BeamMonitor(f"monitor_{nslice:02d}", backend="h5")
        #monitor = elements.Drift(ds=0.0, name="nulldrift")
        print(f'created monitor for nslice:{nslice}, id: {id(monitor)}')
        sim.lattice.extend(
            [
                monitor,
                # Sbend, 90 degree bend radius 1 m
                elements.ExactSbend(np.pi/2, 90.0, name=f"sbend_{nslice:02d}", nslice=nslice),
                monitor,
            ]
        )

        sims[nslice] = sim

    # run them all
    for s in sims:
        sims[s].track_particles()

    for s in sims:
        sims[s].finalize()

    return

if __name__ == "__main__":
    main()
    pass

