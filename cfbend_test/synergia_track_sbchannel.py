#!/usr/bin/env python3
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import synergia
PCONST = synergia.foundation.pconstants

macroparticles = 16
realparticles = 5e10

def get_lattice(f):
    lattice = synergia.lattice.MadX_reader().get_lattice("channel", f)
    return lattice

def create_sim(ref_part):
    sim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(
        ref_part, macroparticles, realparticles
    )
    bunch = sim.get_bunch()
    bunch.checkout_particles()
    lp = bunch.get_particles_numpy()
    lp[:, 0:6] = 0.0
    lp[:11, 0] = 0.001 * np.arange(-5.0, 5.01, 1.0)
    lp[:11, 5] = 5.0e-3
    bunch.checkout_particles()
    return sim

    
def create_propagator(lattice):
    stepper = synergia.simulation.Independent_stepper_elements(1)
    propagator = synergia.simulation.Propagator(lattice, stepper)
    return propagator
    
def reg_diagnostics(nm, sim):
    diag = synergia.bunch.Diagnostics_bulk_track(f"{nm}tracks.h5", 11)
    sim.reg_diag_per_turn(diag)

def runit(lname):
    lattice = get_lattice(f"{lname}channel.madx")
    print('read lattice, length: ', lattice.get_length())

    refpart = lattice.get_reference_particle()

    sim = create_sim(refpart)
    reg_diagnostics(lname, sim)

    propagator = create_propagator(lattice)
    simlog = synergia.utils.parallel_utils.Logger(0,
                            synergia.utils.parallel_utils.LoggerV.INFO_TURN)
    propagator.propagate(sim, simlog, 1)

def main():
    runit("sb")
    runit("nb")
    return
        
if __name__ == "__main__":
    main()

        
