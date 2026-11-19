#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c, eV as qe, epsilon_0
from impactx import ImpactX, distribution, elements, amr

R = 0.001 # Radius of sphere in rest frame
a = 1.5 # factor for extension of test particles
L = 1.0

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = "Gauss3D"
sim.space_charge_gauss_nint = 101
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

myrank = amr.ParallelDescriptor.MyProc()

# load a 250 MeV electron beam
# in a (rest frame) sphere radius of 0.001 m
kin_energy_MeV = 250  # reference energy
bunch_charge_C = 1.0e-9  # used with space charge
npart = 100000  # number of macro particles
Q = bunch_charge_C

#   reference particle
ref = sim.beam.ref
ref.set_species("electron").set_kin_energy_MeV(kin_energy_MeV)

qm_eev = 1.0/(1.0e6 * ref.mass_MeV)

beta = ref.beta
gamma = ref.gamma
beta_gamma = ref.beta_gamma

m_e = ref.mass_MeV

rng = np.random.default_rng()
u = rng.random(npart)
v = rng.random(npart)
w = rng.random(npart)

r = R * u**(1/3)
phi = 2*np.pi*v
theta = np.arccos(2*w-1)

zeroc = np.zeros(npart)
exes = r*np.sin(theta)*np.cos(phi)
wyze = r*np.sin(theta)*np.sin(phi)
# make the bunch spherical in the rest frame
zees = r*np.cos(theta) / (beta_gamma)

sim.beam.add_n_particles(exes, wyze, zees, zeroc, zeroc, zeroc, qm_eev, bunch_charge=bunch_charge_C)

# Add test particles at the edges of the sphere
test_x = np.array([R*a, -R*a, 0, 0, 0, 0])
test_y = np.array([0, 0, R*a, -R*a, 0, 0])
test_t = np.array([0, 0, 0, 0, a*R/beta_gamma, -a*R/beta_gamma])
test_zeros = np.zeros(6)

# Add test particles
sim.beam.add_n_particles(test_x, test_y, test_t, test_zeros, test_zeros, test_zeros, qm_eev, w=test_zeros)

# add beam diagnostics
#monitor = elements.BeamMonitor("monitor", backend="h5")

# design the accelerator lattice)
ns = 1  # number of slices per ds in the element
channel = [
#    monitor,
    elements.ExactDrift(name="drift1", ds=L, nslice=ns),
#    monitor,
]

# assign a fodo segment
sim.lattice.extend(channel)

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
sim.hook["after_period"] = collect_test_particle_data

# run simulation
sim.track_particles()
if myrank == 0:
    np.save(sim.diag_file_prefix + "/test_particle_data.npy", test_particle_data, allow_pickle=True)

# clean shutdown
sim.finalize()

# turn the data in to a dataframe
df = pd.DataFrame(test_particle_data, columns=["turn", "idcpu", "x", "px", "y", "py", "t", "pt"])

# Group by particle ids, without changing their relative order
grouped_by_idcpu = df.groupby("idcpu", sort=True)
particle_ids = df["idcpu"].unique().tolist()
print('number of particles', len(particle_ids))

# print(grouped_by_idcpu)
# for i in range(len(grouped_by_idcpu)):
#     print(grouped_by_idcpu[i])

print(df[df["idcpu"] == particle_ids[0]][["turn", "x", "px"]])
print(df[df["idcpu"] == particle_ids[4]][["turn", "t", "pt"]])

xpx0 = df[df["idcpu"] == particle_ids[0]][["x", "px"]].to_numpy()
print(xpx0)
xpx1 = df[df["idcpu"] == particle_ids[1]][["x", "px"]].to_numpy()

ypy0 = df[df["idcpu"] == particle_ids[2]][["y", "py"]].to_numpy()
ypy1 = df[df["idcpu"] == particle_ids[3]][["y", "py"]].to_numpy()

tpt0 = df[df["idcpu"] == particle_ids[4]][["t", "pt"]].to_numpy()
tpt1 = df[df["idcpu"] == particle_ids[5]][["t", "pt"]].to_numpy()

# calculate what the kicks should be

# E_r in the rest frame at the test location a*R
E_r = 1/(4*np.pi*epsilon_0) * Q/(a*R)**2 # [V]

# proper time for given L
t_prop = L/(beta*c) / gamma
# transverse momentum kick
dp = E_r * qe * t_prop # [ kg m/s ]
dp_mev_c = 1.0e-6 * dp * c/qe
# Normalize by p_ref
dpop_t = dp_mev_c/(m_e * beta_gamma)

print('transverse dp_mev_c: ', dp_mev_c)
print('transverse dp/p: ', dpop_t)

# longitudinal kicks
# calculate energy and momentum for lorentz transform
p_long_rf = dp_mev_c # pc [MeV[
e_long_rf = np.sqrt(dp_mev_c**2 + m_e**2) #  [MeV]

# The particle in +cdt is actually at negative z and has
# it's momentum decreased, so we use -p_long_rf in the lorentz
# transformation
p_long_back_lab = beta_gamma * e_long_rf +  gamma * -dp_mev_c
e_lab_back = gamma * e_long_rf + beta_gamma * -p_long_rf
pt_back_lab = (m_e * gamma - e_lab_back)/(m_e * beta_gamma)
print('pt_back_lab: ', pt_back_lab)
print('tracked long pt back: ', tpt0[1, 1])

e_lab_for = gamma * e_long_rf + beta_gamma * +p_long_rf
pt_for_lab = (m_e * gamma - e_lab_for)/(m_e * beta_gamma)
print('pt_for_lab: ', pt_for_lab)
print('tracked long pt forward: ', tpt1[1, 1])

p_long_forward_lab = beta_gamma * e_long_rf +  gamma * dp_mev_c
e_long_forward_lab =  gamma * e_long_rf + beta_gamma * p_long_rf
pt_forward_lab =  (m_e * gamma - e_long_forward_lab)/(m_e * beta_gamma)

# compare with solver
print()
print('Analytic transverse kick: ', dpop_t)
print('+x kick: ', xpx0[1,1], 'relative diff: ', (xpx0[1, 1] - dpop_t)/(xpx0[1, 1] + dpop_t))

print('-x kick: ', xpx1[1, 1], 'relative diff: ', (xpx1[1, 1] + dpop_t)/(dpop_t - xpx1[1, 1]))

print('+y kick: ', ypy0[1,1], 'relative diff: ', (ypy0[1, 1] - dpop_t)/(ypy0[1, 1] + dpop_t))

print('-y kick: ', ypy1[1,1], 'relative diff: ', (ypy1[1, 1] + dpop_t)/(dpop_t - ypy1[1, 1]))

print()
print('Analytic backward PT: ', pt_back_lab)
print('backward PT: ', tpt0[1, 1], 'relative diff: ', (pt_back_lab - tpt0[1,  1])/(pt_back_lab + tpt0[1, 1]) )

print()
print('Analytic forward PT: ', pt_forward_lab)
print('forward PT: ', tpt1[1,1], 'relative diff: ', (pt_forward_lab - tpt1[1,  1])/(pt_forward_lab + tpt1[1, 1]) )

