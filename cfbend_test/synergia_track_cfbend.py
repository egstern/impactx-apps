#!/usr/bin/env python3
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import synergia
PCONST = synergia.foundation.pconstants

macroparticles = 24
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

    offs = 0.005
    dpop = 5.0e-3

    # all the particles will be at 0 for this tes
    lp = sim.get_bunch(0, 0).get_particles_numpy()
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

        lp[:, 5] = dpop

    sim.get_bunch(0, 0).checkin_particles()

    bunch.checkout_particles()
    return sim

    
def create_propagator(lattice):
    stepper = synergia.simulation.Independent_stepper_elements(1)
    propagator = synergia.simulation.Propagator(lattice, stepper)
    return propagator
    
def reg_diagnostics(nm, sim):
    diag = synergia.bunch.Diagnostics_bulk_track(f"cftracks.h5", 24)
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
    return
        
if __name__ == "__main__":
    main()

        
