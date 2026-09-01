#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Eric G. Stern, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-


from impactx import Config, ImpactX, distribution, elements

from booster_apertures import DMAG_aperture

sim = ImpactX()

# set numerical parameters and IO control
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# load a 0.8 GeV proton beam with an initial
# unnormalized rms emittance of 2 nm
kin_energy_MeV = 800.0  # reference energy 800 MeV proton
bunch_charge_C = 1.0e-9  # used with space charge
npart = 50000  # number of macro particles

#   reference particle
ref = sim.beam.ref
ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)

#   particle bunch
distr = distribution.Waterbag(
    lambdaX=6.0e-2,
    lambdaY=6.0e-2,
    lambdaT=0.001,
    lambdaPx=4.0e-6,
    lambdaPy=4.0e-6,
    lambdaPt=2.0e-10,
    muxpx=0.0,
    muypy=0.0,
    mutpt=0.0,
)
sim.add_particles(bunch_charge_C, distr, npart)

# add beam diagnostics
monitor = elements.BeamMonitor("monitor", backend="h5")


# design the accelerator lattice)
ns = 1  # number of slices per ds in the element
channel = [
    monitor,
    DMAG_aperture,
    monitor,
]

sim.lattice.extend(channel)

# run simulation
sim.track_particles()

# clean shutdown
sim.finalize()
