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
npart = 10000  # number of macro particles
emit_x = 8.0e-6 # 8 pi mm-mr sensible 90% emittance
emit_y = 8.0e-6
init_std_dpop = 1.0e-4

myrank = MPI.COMM_WORLD.rank

# Read the lattice
lattice = synergia.lattice.MadX_reader().get_lattice('model', 'foborodobo32.madx')

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

synergia.simulation.Lattice_simulator.tune_circular_lattice(lattice)
synergia.simulation.Lattice_simulator.CourantSnyderLatticeFunctions(lattice)
synergia.simulation.Lattice_simulator.calc_dispersions(lattice)
lf = lattice.get_elements()[-1].lf
beta_x = lf.beta.hor
alpha_x = lf.alpha.hor
beta_y = lf.beta.ver
alpha_y = lf.alpha.ver
psi_x = lf.psi.hor
psi_y = lf.psi.ver
disp_x = lf.dispersion.hor
dprime_x = lf.dPrime.hor

if myrank == 0:
    print('lattice functions')
    print('beta_x: ', beta_x)
    print('alpha_x', alpha_x)
    print('beta_y: ', beta_y)
    print('alpha_y: ', alpha_y)
    print('dispersion_x: ', disp_x)
    print('dispersion prime_x: ', dprime_x)
    print()
    

# Get original tunes and chromaticities
rf_freq = synergia.simulation.Lattice_simulator.get_rf_frequency(lattice)
(xtune, ytune, orbit_cdt) = synergia.simulation.Lattice_simulator.calculate_tune_and_cdt(lattice)
chrom = synergia.simulation.Lattice_simulator.get_chromaticities(lattice)
hchrom = chrom.horizontal_chromaticity
vchrom = chrom.vertical_chromaticity
if myrank == 0:
    print("Orbit length: ", beta*orbit_cdt, "m")
    print("RF frequency: ", rf_freq, "Hz")
    print("horizontal chromaticity: ", hchrom)
    print("vertical chromaticity: ", vchrom)
    print()


if myrank == 0:
    print("generating matched particle distribution")
    print("num particles: ", npart)
    print("initial x emittance: ", emit_x, " pi m-R (90%)")
    print("initial y emittance: ", emit_y, " pi m-R (90%)")
    print("initial std dp/p: ", init_std_dpop)


# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# load a 800 MeV KE proton as specified in the MAD-X file


# calculate parameters for initial distribution
stdx = np.sqrt(emit_x/(4*beta_x) + init_std_dpop**2 * disp_x**2)
stdy = np.sqrt(emit_y/(4*beta_y))

map = synergia.simulation.Lattice_simulator.get_linear_one_turn_map(lattice)
print('one turn map')
print(map)
print()
# correlation matrix based in stdx, stdy and std_dpop
corr_matrix = synergia.bunch.get_correlation_matrix(map, stdx, stdy, init_std_dpop, beta, (0,2, 5))

print('corr_matrix')
print(corr_matrix)

print('sqrt(cm[0,0]: ', np.sqrt(corr_matrix[0,0]))
print('sqrt(cm[2,2]: ', np.sqrt(corr_matrix[2,2]))
print('sqrt(cm[4,4]: ', np.sqrt(corr_matrix[4,4]))
print('sqrt(cm[5,5]: ', np.sqrt(corr_matrix[5,5]))
print('beta again?: ', beta)

if myrank == 0:
    print("Bunch parameters for generation")
    print("stdx: ", stdx)
    print("stdy: ", stdy)
    print("std s: ", np.sqrt(corr_matrix[4, 4])*beta)
    print("std dp/p: ", np.sqrt(corr_matrix[5, 5]))

dist = synergia.foundation.Random_distribution(12345679, myrank)
                                                    
bunchsim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(
    refpart, npart, bunch_charge_C)

bunch = bunchsim.get_bunch(0, 0)
means = np.zeros((6), dtype='d')
synergia.bunch.populate_6d(dist, bunch, means, corr_matrix)
bunch.checkin_particles()

# Create the ImpactX reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(refpart.get_charge())
ref.set_mass_MeV(refpart.get_mass()*1000.0) # MeV units ickk!
kin_energy = (refpart.get_total_energy() - refpart.get_mass())*1000.0
ref.set_kin_energy_MeV(kin_energy) #MeV units ickk!

bunch.checkout_particles()
local_part = bunch.get_particles_numpy()

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

# don't need the synergia bunch anymore. Delete it
# so there aren't any Kokkos remnants hanging around
del dist
del corr_matrix
del local_part
del bunch
del bunchsim

# charge to mass ratio
qm_eev = 1.0/(1.0e-9*mp)

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)

# insert the converted MAD-X->Synergia lattice
sim.lattice.extend(syn2_to_impactx(lattice))

print('impactx lattice:')
print(sim.lattice)

# run simulation
sim.periods=1
sim.track_particles()

# clean shutdown
sim.finalize()
