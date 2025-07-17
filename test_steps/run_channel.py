#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Marco Garten, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-


import sys, os
import mpi4py.MPI as MPI
import numpy as np
from scipy import constants
import synergia
from  syn2_to_impactx import syn2_to_impactx

import amrex.space3d as amr
from impactx import ImpactX, Config, distribution
c = constants.c
e = constants.e
mp = synergia.foundation.pconstants.mp

sim = ImpactX()

bunch_charge_C = e * 0.5e10
npart = 24  # number of macro particles
emit_x = 8.0e-6 # 8 pi mm-mr sensible 90% emittance
emit_y = 8.0e-6
init_std_dpop = 1.0e-4

myrank = MPI.COMM_WORLD.rank

# Read the lattice
lattice = synergia.lattice.MadX_reader().get_lattice('model', 'channel.madx')

refpart = lattice.get_reference_particle()
if myrank == 0:
    energy = refpart.get_total_energy()
    momentum = refpart.get_momentum()
    gamma = refpart.get_gamma()
    beta = refpart.get_beta()

    print("Beam parameters")
    print("energy: ", energy)
    print("momentum: ", momentum)
    print("gamma: ", gamma)
    print("beta: ", beta)
    print()


if True:
    # generate particles around a cylinder at 15 degree angles
    s15 = (np.sqrt(6) - np.sqrt(2))/4
    c15 = (np.sqrt(6) + np.sqrt(2))/4
    s30 = 0.5
    c30 = np.sqrt(3)/2
    s45 = np.sqrt(2)/2
    c45 = s45
    s60 = c30
    c60 = s30
    s75 = c15
    c75 = s15

    offs = 0.001

    lp = np.zeros((24, 6))
    # all the particles will be at 0 for this tes
    lp[:, 0:6] = 0.0
    lp[0, 0] = offs
    lp[1, 0] = offs * c15
    lp[1, 2] = offs * s15
    lp[2, 0] = offs * c30
    lp[2, 2] = offs * s30
    lp[3, 0] = offs * c45
    lp[3, 2] = offs * s45
    lp[4, 0] = offs * c60
    lp[4, 2] = offs * s60
    lp[5, 0] = offs * c75
    lp[5, 2] = offs * s75

    # duplicate it rotated 90 degrees
    for i in range(6):
        # y->-x, x->y
        lp[i+6, 0] = -lp[i, 2]
        lp[i+6, 2] = lp[i, 0]
    # no duplicate everything flipped 180 degrees
    for i in range(12):
        lp[i+12, 0] = -lp[i, 0]
        lp[i+12, 2] = -lp[i, 2]

    pass


# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# load a 800 MeV KE proton as specified in the MAD-X file


# Create the ImpactX reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(refpart.get_charge())
ref.set_mass_MeV(refpart.get_mass()*1000.0) # MeV units ickk!
kin_energy = (refpart.get_total_energy() - refpart.get_mass())*1000.0
ref.set_kin_energy_MeV(kin_energy) #MeV units ickk!

local_part = lp

# convert from dp/p to dE/p
pt = -(np.sqrt( (momentum*(1 + local_part[:, 5]) )**2 + mp**2) - energy)/momentum
local_part[:, 5] = pt[:]

# Load into ImpactX particle container
pc = sim.particle_container()

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

for p_dx in local_part[:,0]:
    dx_podv.push_back(p_dx)
for p_dy in local_part[:,2]:
    dy_podv.push_back(p_dy)
for p_dt in local_part[:,4]:
    dt_podv.push_back(p_dt)
for p_dpx in local_part[:,1]:
    dpx_podv.push_back(p_dpx)
for p_dpy in local_part[:,3]:
    dpy_podv.push_back(p_dpy)
for p_dpt in local_part[:,5]:
    dpt_podv.push_back(p_dpt)

# charge to mass ratio
qm_eev = 1.0/(1.0e-9*mp)

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)

# insert the converted MAD-X->Synergia lattice
IX_lattice = syn2_to_impactx(lattice)
sim.lattice.extend(IX_lattice)

print('impactx lattice:')
print(sim.lattice)
print(dir(sim.lattice))
for e in sim.lattice:
    print(e)
    print()

# Cannot I address elements by index?
elem2 = None
i = 0
for e in sim.lattice:
    if i == 1:
        elem2 = e
        break
    else:
        i = i + 1

print('second lattice element2: ', elem2)

# is this the same object as what I put into the sim?
print('same? ', IX_lattice[1] is elem2)

# run simulation
sim.periods=1
sim.track_particles()

# clean shutdown
sim.finalize()
