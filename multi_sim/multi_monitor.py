#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

from impactx import ImpactX, distribution, elements

sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = False

# domain decomposition & space charge mesh
sim.init_grids()

# load a 2 GeV electron beam with an initial
# unnormalized rms emittance of 2 nm
kin_energy_MeV = 2.0e3  # reference energy
bunch_charge_C = 1.0e-9  # used with space charge
npart = 10000  # number of macro particles

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(-1.0).set_mass_MeV(0.510998950).set_kin_energy_MeV(kin_energy_MeV)

#   particle bunch
distr = distribution.Waterbag(
    lambdaX=3.9984884770e-5,
    lambdaY=3.9984884770e-5,
    lambdaT=1.0e-3,
    lambdaPx=2.6623538760e-5,
    lambdaPy=2.6623538760e-5,
    lambdaPt=2.0e-3,
    muxpx=-0.846574929020762,
    muypy=0.846574929020762,
    mutpt=0.0,
)
sim.add_particles(bunch_charge_C, distr, npart)

# add beam diagnostics
monitor1 = elements.BeamMonitor("monitor1", backend="h5")
monitor2 = elements.BeamMonitor("monitor2", backend="h5")

# design the accelerator lattice)
ns = 25  # number of slices per ds in the element
fodo = [
    elements.Drift(name="drift1", ds=0.25, nslice=ns),
    elements.Quad(name="quad1", ds=1.0, k=1.0, nslice=ns),
    monitor1,
    elements.Drift(name="drift2", ds=0.5, nslice=ns),
    elements.Quad(name="quad2", ds=1.0, k=-1.0, nslice=ns),
    monitor2,
    elements.Drift(name="drift3", ds=0.25, nslice=ns),
]
# assign a fodo segment
sim.lattice.extend(fodo)

# run simulation
sim.periods=10
sim.track_particles()

# clean shutdown
sim.finalize()
