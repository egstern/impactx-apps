#!/usr/bin/env python3

#
# Copyright 2022-2023 ImpactX contributors
# Authors: Eric G. Stern, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np

from scipy.constants import c, eV, m_p, pi

from get_lattice import get_lattice
from booster_rf import *
pip2ramp = PIP2ramp()

from booster_momentum import *

from booster_set_rf import set_rf

from impactx import ImpactX, distribution, elements, twiss, synmadx

from booster_accel_options import Options
from scipy.constants import c

from mpi4py import MPI
myrank = MPI.COMM_WORLD.rank

#========================================================================

opts = Options()

# write out RF and energy status turn-by-turn on rank 0
if myrank == 0:
    runstatus = open('runstatus.txt', 'w')
    print("turn time gamma V phase", file=runstatus)
    pass

# Update RF cavities for next turn
def update_rf_cavities_next_turn(sim):
    ref = sim.beam.ref
    booster_momentum = Booster_momentum(ref)
    current_beta = ref.beta
    current_gamma = ref.gamma
    above_transition = current_gamma > opts.gamma_tr
    mass_MeV = ref.mass_MeV
    current_E = mass_MeV * ref.gamma
    current_time = ref.t/c
    lattice_length = sum(elem.ds for elem in sim.lattice)
    delta_t = lattice_length/(current_beta*c)
    E_next_turn = booster_momentum.e_vs_t(current_time + delta_t)
    new_freq = opts.harmonic_number * current_beta * c/lattice_length
    required_energy_gain = E_next_turn - current_E
    current_V = pip2ramp.get_rf_voltage_by_time(current_time)
    if required_energy_gain/current_V > 1.0:
        print(f"Oh Noooo! Energy gain {required_energy_gain} is larger than current voltage {current_V}")
        # use maximum
        phase_needed = np.pi/2
    else:
        phase_needed = np.arcsin(required_energy_gain/current_V)
    set_rf(sim, current_V, freq=new_freq, phaseR=phase_needed, above_transition=above_transition)

    print(sim.tracking_period, current_time, current_gamma, current_V,
          phase_needed, file=runstatus, flush=True)

    return


#========================================================================

# main driver of the simulation
def main():

    opts = Options()

    sim = ImpactX()

    # set numerical parameters and IO control
    sim.space_charge = False
    # sim.diagnostics = False  # benchmarking
    sim.slice_step_diagnostics = True

    # domain decomposition & space charge mesh
    sim.init_grids()
    
    # Read lattice and get bucket length
    ix_lattice = get_lattice()
    lattice_length = sum(elem.ds for elem in ix_lattice)
    print("lattice length: ", lattice_length)

    bucket_length = lattice_length/opts.harmonic_number
    print("bucket_length: ", bucket_length)
    
    # Set up reference particle
    init_energy = opts.injection_energy
    ref = sim.beam.ref
    ref.set_species("proton")
    ref.set_kin_energy_MeV(init_energy)

    # set periodic particle bucket boundary
    sim.particle_bc = "periodic"
    sim.beam.set_bucket_length(bucket_length)

    # element to read in particles
    particle_source = elements.Source(distribution="openPMD", openpmd_path=opts.particles_file, active_once=True, name="particles")
    monitor = elements.BeamMonitor("monitor.h5", backend="h5")

    sim.lattice.clear()
    sim.lattice.append(particle_source)
    sim.lattice.extend(ix_lattice)
    sim.lattice.append(monitor)

    # Set up for updating RF and frequency
    sim.hook["before_period"] = update_rf_cavities_next_turn

    sim.periods = opts.turns
    sim.verbose = 0
    sim.track_particles()
    
    sim.finalize()

    return


#========================================================================

if __name__ == "__main__":
    main()
