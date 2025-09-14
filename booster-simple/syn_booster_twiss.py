#!/usr/bin/env python3

# this script is for synergia2
import sys, os
import numpy as np
import synergia


def main(lattice_file, line):
    lattice = synergia.lattice.MadX_reader().get_lattice(line, lattice_file)
    beta = lattice.get_reference_particle().get_beta()
    gamma = lattice.get_reference_particle().get_gamma()
    lattice.set_all_string_attribute("extractor_type", "chef")
    prefix = os.path.split(lattice_file)[0]
    if not prefix:
        prefix="."
    basefile = os.path.splitext(os.path.split(lattice_file)[1])[0]
    twissout = prefix + '/syn_' + basefile+'_twiss.out'
    stepper = synergia.simulation.Independent_stepper(lattice, 1, 1)
    lattice_simulator = stepper.get_lattice_simulator()
    lfoutput = open(twissout, 'w')
    print("# NAME  S   L  BETX  ALFX  MUX   BETY   ALFY   MUY", file=lfoutput)
    for elem in lattice.get_elements():
        lf = lattice_simulator.get_lattice_functions(elem)
        print(elem.get_name(), lf.arc_length, elem.get_length(), lf.beta_x, lf.alpha_x, lf.psi_x, lf.beta_y, lf.alpha_y, lf.psi_y, file=lfoutput)

    lfoutput.close()

    print('beta: ', beta)
    print('gamma: ', gamma)

    chrom_x = lattice_simulator.get_horizontal_chromaticities()
    chrom_y = lattice_simulator.get_vertical_chromaticities()
    print('horizontal chromaticities: ', chrom_x)
    print('horizontal chromaticity to compare MADX: ', chrom_x[0]/beta)
    print('vertical chromaticities: ', chrom_y)
    print('vertical chromaticity to compare to MADX: ', chrom_y[0]/beta)
    compact = lattice_simulator.get_momentum_compaction()
    print('momentum compaction: ', compact)
    print('momentum compaction factor to compare to MADX: ', compact/beta)
    slip = lattice_simulator.get_slip_factor()
    print('slip factor: ', slip)
    print(' alpha - 1/gamma**2: ', compact-1/gamma**2)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
