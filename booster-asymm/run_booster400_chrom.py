#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Eric G. Stern, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

from booster400_adjusted_IX_lattice import get_lattice
from scipy.constants import c, eV, m_p, pi
from numpy import sqrt, arange, linspace, zeros, ones, save as npsave

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
sim.diag_file_prefix = "diags_chrom"
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

# Test particles distributed in PT from -0.001 to 0.001
sigma_x = sqrt( (beta_x * emit_x) + (disp_x**2 * emit_z/beta_z) )
sigma_y = sqrt( beta_y * emit_y )

ptmin = -0.00001
ptmax = 0.00001

if myrank == 0:
    print("expected sigma_x: ", sigma_x)
    print("expected sigma_y: ", sigma_y)

ntest_per_plane = 25
pttest = linspace(ptmin, ptmax, ntest_per_plane)
#xtest = arange(ntest_per_plane) * 5 * sigma_x/ntest_per_plane
# central particle will not have any offset so give it a little
xtest = ones(ntest_per_plane) * 1.0e-8
#ytest = arange(ntest_per_plane) * 5 * sigma_y/ntest_per_plane
ytest = ones(ntest_per_plane) * 1.0e-8
zero_arr  = zeros(ntest_per_plane)

qm_eev = 1.0/(mp_mev * 1.0e-6)
if myrank == 0:
    sim.beam.add_n_particles(xtest, zero_arr, zero_arr, zero_arr, zero_arr, pttest, qm_eev, bunch_charge=0.0)
    sim.beam.add_n_particles(zero_arr, ytest, zero_arr, zero_arr, zero_arr, pttest, qm_eev, bunch_charge=0.0), 

# add beam diagnostics
monitor0 = elements.BeamMonitor("monitor", backend="bp5", encoding='v', period_sample_intervals=1000000, particles=False)
monitor1 = elements.BeamMonitor("monitor", backend="bp5", encoding='v', particles=False)

zerodrift = elements.Drift(ds=0.0)

sim.verbose = verbosity

# initial 0 length drift is to force initial phase wrapping
push(sim.beam, zerodrift)

# get the Booster lattice, turn off rf cavities
booster_lattice = get_lattice()
for elem in booster_lattice:
    if elem.to_dict()['type'] == "ShortRF":
        elem.V = 0.0
    
sim.lattice.clear()
sim.lattice.append(monitor0)
sim.lattice.extend(booster_lattice)
sim.lattice.append(monitor1)

test_particle_data = []
def collect_test_particle_data(sim):

    turn = sim.tracking_period

    beam = sim.beam.to_df(local=False)
    # Filter on particle weight (collect test particles only)
    if beam is not None:
        for row in beam[beam["weighting"] == 0.0].itertuples():
            # collect test particle data
            test_particle_data.append([turn, row.idcpu, row.position_x, row.momentum_x, row.position_y, row.momentum_y, row.position_t, row.momentum_t])

sim.hook["before_period"] = collect_test_particle_data

# run simulation
sim.periods = turns
sim.track_particles()

if myrank == 0:
    npsave(sim.diag_file_prefix + "/test_particle_data.npy", test_particle_data, allow_pickle=True)

# clean shutdown
sim.finalize()
