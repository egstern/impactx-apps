#!/usr/bin/env python
import sys
import os
import numpy as np
import synergia
import synergia.simulation as SIM
from channel_options import opts
ET = synergia.lattice.element_type
MT = synergia.lattice.marker_type
#####################################


################################################################################

def get_lattice():
    # read the lattice in from a MadX sequence file
    lattice = synergia.lattice.MadX_reader().get_lattice("model","channel.madx")
    lattice.set_all_string_attribute("extractor_type", "libff")
    return lattice

################################################################################



################################################################################

def main():

    logger = synergia.utils.Logger(0)

    lattice = get_lattice()
    print('Read lattice, length = {}, {} elements'.format(lattice.get_length(), len(lattice.get_elements())), file=logger)

    # assume the lattice sets the reference particle
    refpart = lattice.get_reference_particle()

    energy = refpart.get_total_energy()
    momentum = refpart.get_momentum()
    gamma = refpart.get_gamma()
    beta = refpart.get_beta()

    print("energy: ", energy, file=logger)
    print("momentum: ", momentum, file=logger)
    print("gamma: ", gamma, file=logger)
    print("beta: ", beta, file=logger)

    f = open("channel_lattice.out", "w")
    print(lattice, file=f)
    f.close()

    num_bunches = opts.num_bunches
    bucket_length = lattice.get_length()/opts.harmon
    
    macroparticles = opts.macroparticles
    real_particles = opts.real_particles
    print("macroparticles: ", macroparticles, file=logger)
    print("real_particles: ", real_particles, file=logger)

    sim = synergia.simulation.Bunch_simulator.create_bunch_train_simulator(
        refpart, macroparticles, real_particles, num_bunches, bucket_length)

    if opts.periodic:
        sim.set_longitudinal_boundary(
                synergia.bunch.LongitudinalBoundary.periodic, bucket_length)

    comm = synergia.utils.parallel_utils.Commxx()

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

    sim.get_bunch(0, 0).checkin_particles()

    steps = opts.steps
    turns = opts.turns

    grid = [opts.gridx, opts.gridy, opts.gridz]

    # space charge
    if opts.spacecharge:
        # space charge active
        comm_group_size = opts.comm_group_size

        if ops.spacecharge == "3dopen-hockney":
            sc_ops = synergia.collective.Space_charge_3d_open_hockney_options(gridx, gridy, gridz)
            sc_ops.comm_group_size = comm_group_size

        elif solver == "2dopen-hockney":
            sc_ops = synergia.collective.Space_charge_2d_open_hockney_options(gridx, gridy, gridz)
            sc_ops.comm_group_size = comm_group_size

        #elif solver == "2dbassetti-erskine":
        #    space_charge = synergia.collective.Space_charge_2d_bassetti_erskine()

        elif solver == "rectangular":
            space_charge = synergia.collective.Space_charge_rectangular(grid, opts.pipe_size)
            sc_ops.comm_group_size = comm_group_size

        else:
            sys.stderr.write("foborodobo32.py: solver must be either 3dopen-hockney, 2dopen-hockney, or rectangular\n")
            sys.exit(1)

        stepper = synergia.simulation.Split_operator_stepper(sc_ops, steps)

    else:
        # space charge not active

        if opts.stepper == "splitoperator":
            sc_ops = synergia.collective.Dummy_CO_options()
            stepper = synergia.simulation.Split_operator_stepper(sc_ops, steps)

        #elif opts.stepper == "independent":
        #    stepper = synergia.simulation.Independent_stepper(steps)

        elif opts.stepper == "elements":
            stepper = synergia.simulation.Independent_stepper_elements(steps)

        else:
            sys.stderr.write("mi.py: stepper must be either splitopertor,independent, or elements\n")
            sys.exit(1)

    # propagator
    propagator = synergia.simulation.Propagator(lattice, stepper)

    # diagnostics for the bunches
    for bunch_num in range(num_bunches):

        # diagnostics always on
        sim.reg_diag_per_turn(synergia.bunch.Diagnostics_full2("diag_b%03d.h5"%bunch_num),
                              bunch_idx = bunch_num)

        if opts.step_diag:
            sim.reg_diag_per_step(synergia.bunch.Diagnostics_full2("stepdiag_b%03d.h5"%bunch_num),
                    bunch_idx = bunch_num)

        if opts.particles:
            if opts.particles_period == 0:
                sim.reg_diag_per_turn(synergia.bunch.Diagnostics_particles("particles_b%03d.h5"%bunch_num), bunch_idx = bunch_num)
            else:
                turn_list = list(range(0, turns, opts.particles_period))
                if turns-1 not in turn_list:
                    turn_list.append(turns-1)
                    sim.reg_diag_turn_listed(synergia.bunch.Diagnostics_particles("mi_particles_b%03d.h5"%bunch_num), 
                                             bunch_idx = bunch_num, turns = turn_list)

        # enable track saving
        # each processor will save tracks/proc tracks
        if opts.tracks:
            trkfile = 'tracks_b%03d.h5'%bunch_num
            sim.reg_diag_per_step(synergia.bunch.Diagnostics_bulk_track(trkfile, opts.tracks), bunch_idx = bunch_num)

    # max simulation turns
    sim.set_max_turns(opts.turns)

    # logs
    #simlog = synergia.utils.parallel_utils.Logger(0, synergia.utils.parallel_utils.LoggerV.INFO_STEP)
    simlog = synergia.utils.parallel_utils.Logger(0, synergia.utils.parallel_utils.LoggerV.INFO_TURN)
    screen = synergia.utils.parallel_utils.Logger(0, synergia.utils.parallel_utils.LoggerV.DEBUG)

    # propagate
    propagator.propagate(sim, simlog, turns)

if __name__ == "__main__":
    main()

