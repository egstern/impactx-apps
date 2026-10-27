#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Eric G. Stern, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

from booster_asymm_IX_lattice import get_lattice
from scipy.constants import c, eV, m_p, pi
from numpy import sqrt, arange, zeros

from impactx import ImpactX, distribution, elements, twiss, push
from impactx import amr


mp_mev = 1.0e-6 * m_p * c**2 / eV
total_Booster_charge = 6.7e12  # PIP-II full Booster
active_buckets = 81  # 81 out of 84 buckets full
harmonic_number = 84

turns = 1024

# Geometric emittances from Synergia 400 MeV Booster simulation
# booster400-SCrect.02 at turn 50

emit_x = 2.08e-6
emit_y = 1.73e-6
emit_z = 0.00161 # (ct-dp/p)
beta_z = 879 # sig-t/sig-dpop
#
# these lattice functions are calculated with Synergia during
# construction of the booster-asymm lattice
alpha_x = -0.03374203811363915
beta_x = 33.432491467233
alpha_y = 0.011403474422954477
beta_y = 5.2633267186119905

disp_x = 3.071236819185704
disp_px = 0.003057679536025426 # /(dpop)

verbosity = 0
gen_diagnostics = False
step_diagnostics = False
SC_solver = None

sim = ImpactX()

# domain decomposition & space charge mesh
sim.init_grids()

myrank = amr.ParallelDescriptor.MyProc()

# set numerical parameters and IO control
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.diagnostics = gen_diagnostics
sim.slice_step_diagnostics = step_diagnostics

# set the bucket length so that particles can be phase wrapped
booster_lattice = get_lattice()
total_s = sum(element.ds for element in booster_lattice)
print(f"Read Booster lattice, {len(booster_lattice)} elements, len = {total_s}")

bucket_length = total_s / harmonic_number
print("bucket_length: ", bucket_length)
sim.particle_container().set_bucket_length(bucket_length)

# set longitudinal particle boundary to periodic
sim.particle_bc = "periodic"

# load a 400 MeV proton beam approximately the beam captured in
# 400 MeV Booster operations

kin_energy_MeV = 400.1545611624435  # reference energy 400+ MeV
bunch_charge_C = eV * total_Booster_charge / active_buckets  # used with space charge
#npart = 1024*128  # number of macro particles
npart = 1024  # number of macro particles

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)

distr = distribution.Gaussian(
    **twiss(
        beta_x=beta_x,
        alpha_x=alpha_x,
        emitt_x=emit_x,
        beta_y=beta_y,
        alpha_y=alpha_y,
        emitt_y=emit_y,
        emitt_t=emit_z * ref.beta,
        beta_t=beta_z / ref.beta,
        dispersion_x=disp_x / ref.beta,
        dispersion_px=disp_px / ref.beta,
    )
)

sim.add_particles(bunch_charge_C, distr, npart)

# Test particles
sigma_x = sqrt( (beta_x * emit_x) + (disp_x**2 * emit_z/beta_z) )
sigma_y = sqrt( beta_y * emit_y )

if myrank == 0:
    print("expected sigma_x: ", sigma_x)
    print("expected sigma_y: ", sigma_y)

ntest_per_plane = 25
xtest = arange(ntest_per_plane) * 5 * sigma_x/ntest_per_plane
xtest[0] = xtest[0] + 1.0e-8
ytest = arange(ntest_per_plane) * 5 * sigma_y/ntest_per_plane
ytest[0] = ytest[0] + 1.0e-8
zero_arr  = zeros(ntest_per_plane)

qm_eev = 1.0/(mp_mev * 1.0e-6)
if myrank == 0:
    sim.beam.add_n_particles(xtest, zero_arr, zero_arr, zero_arr, zero_arr, zero_arr, qm_eev, bunch_charge=0.0)
    sim.beam.add_n_particles(zero_arr, ytest, zero_arr, zero_arr, zero_arr, zero_arr, qm_eev, bunch_charge=0.0), 

# add beam diagnostics
monitor0 = elements.BeamMonitor("monitor", backend="bp5", encoding='v', period_sample_intervals=1000000)
monitor1 = elements.BeamMonitor("monitor", backend="bp5", encoding='v')

zerodrift = elements.Drift(ds=0.0)

sim.verbose = verbosity

# initial 0 length drift is to force initial phase wrapping
push(sim.beam, zerodrift)

sim.lattice.clear()
sim.lattice.append(monitor0)
sim.lattice.extend(get_lattice())
sim.lattice.append(monitor1)

# run simulation
sim.periods = turns
sim.track_particles()

# clean shutdown
sim.finalize()
