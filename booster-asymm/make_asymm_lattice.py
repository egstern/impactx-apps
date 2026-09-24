#!/usr/bin/env python
import sys
import os
import numpy as np
import synergia
import synergia.simulation as SIM
from syn2_to_impactx import syn2_to_impactx
from impactx import elements, synmadx
import h5py

import pandas as pd

ET = synergia.lattice.element_type
MT = synergia.lattice.marker_type

class opts:
    def __init__(self):
        return
    enable_rf = True
    rf_volt = 0.0002
    xtune = 0.71
    ytune = 0.81
    set_xchrom = -8.0
    set_ychrom = -8.0

#####################################

from mpi4py import MPI

myrank = MPI.COMM_WORLD.rank

DEBUG = True
lattice_file = "sbbooster400.madx"
lattice_line = "booster"
harmonic_number = 84

#####################################


# quick and dirty twiss parameter calculator from 2x2 courant-snyder map array
def map2twiss(csmap):
    cosmu = 0.5 * (csmap[0,0]+csmap[1,1])
    asinmu = 0.5*(csmap[0,0]-csmap[1,1])

    if abs(cosmu) > 1.0:
        raise RuntimeError("map is unstable")

    mu =np.arccos(cosmu)

    # beta is positive
    if csmap[0,1] < 0.0:
        mu = 2.0 * np.pi - mu

    beta = csmap[0,1]/np.sin(mu)
    alpha = asinmu/np.sin(mu)
    tune = mu/(2.0*np.pi)

    return (alpha, beta, tune)


################################################################################

def set_adjust_markers(lattice):
    for elem in lattice.get_elements():
        # focussing quads are in the cps with name qsxxN
        # defocussing quads in  the cpl with name qlxxAN
        # focussing sextupoles are in the cps with name X2sxsxx
        # defocussing quads in  the cpl with name sxlxx
        if elem.get_name() == "qsxx":
            elem.set_marker(MT.h_tunes_corrector)
            #print('short quad corrector')
        elif elem.get_name() == "qlxx":
            elem.set_marker(MT.v_tunes_corrector)
            #print('long quad corrector')
        elif elem.get_name() == "sxsxx":
            elem.set_marker(MT.h_chrom_corrector)
            #print('short chrom corrector')
        elif elem.get_name() == "sxlxx":
            elem.set_marker(MT.v_chrom_corrector)
            #print('long chrom corrector')

#######################################################

def get_lattice():
    # read the lattice in from a MadX sequence file
    lattice = synergia.lattice.MadX_reader().get_lattice(lattice_line, lattice_file)
    lattice.set_all_string_attribute("extractor_type", "libff")
    
    set_adjust_markers(lattice)

    # increase strength of first focussing magnet by 1%
    foundit = False
    for elem in lattice.get_elements():
        if elem.get_type() == ET.sbend and elem.get_double_attribute('k1') > 0.0:
        # found one
            foundit = True
            k1 = elem.get_double_attribute('k1')
            elem.set_double_attribute('k1', k1*1.01)
            break

    if not foundit:
        raise RuntimeError("Can't find the first focussing magnet")

    return lattice

################################################################################

#-------------------------------------------------------------------------------

# set the voltage and tune the lattice
# voltage is total voltage in GV

def set_rf(lattice, voltage, harmno, bunch_phase_offset, phase, above_transition=False, logger=None):

    # above transition, phase needs to be (pi - phase) for longitudinal stability
    if above_transition:
        phase_set = np.pi - phase
    else:
        phase_set = phase

    # Offset phase by how far the bunch has shifted
    phase_set = phase_set + bunch_phase_offset

    if DEBUG and logger:
        print('setrf: lattice: ', id(lattice), file=logger)
        print('set_rf: voltage=', voltage, ', harmno = ', harmno, ', phase = ', phase_set, end='', file=logger)
    # count RF cavities
    cavities = 0
    for elem in lattice.get_elements():
        if elem.get_type() == synergia.lattice.element_type.rfcavity:
            cavities = cavities + 1
    if DEBUG and logger:
        print(' for ', cavities, ' cavities', file=logger)

    # Set the RF cavity voltage
    for elem in lattice.get_elements():
        if elem.get_type() == synergia.lattice.element_type.rfcavity:
            elem.set_double_attribute('volt', 1000*voltage/cavities)
            elem.set_double_attribute('lag', phase_set/(2*np.pi))
            elem.set_double_attribute('harmon', harmno)

    synergia.simulation.Lattice_simulator.tune_circular_lattice(lattice)

    for elem in lattice.get_elements():
        if DEBUG and logger and elem.get_type() == synergia.lattice.element_type.rfcavity:
            print('set_rf: ', elem, file=logger)
            break

    return lattice


#-------------------------------------------------------------------------------

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

    if opts.enable_rf:
        lattice_with_rf = set_rf(lattice, voltage=opts.rf_volt,
                                 harmno=harmonic_number, bunch_phase_offset=0,
                                 phase=0.0, above_transition=False)

    f = open("booster_lattice.out", "w")
    print(lattice, file=f)
    f.close()

    lattice.export_madx_file('booster_lattice.madx', sanitize=True)

    # I'm only reading the lattice to adjust it for asymmetry

    comm = synergia.utils.parallel_utils.Commxx()

    map = SIM.Lattice_simulator.get_linear_one_turn_map(lattice)

    print('map:', file=logger)
    print(np.array2string(map, max_line_width=200), file=logger)

    [l, v] = np.linalg.eig(map)
    print("eigenvalues: ", file=logger)
    for z in l:
        print("|z|: ", abs(z), " z: ", z, " tune: ", np.log(z).imag/(2.0*np.pi), file=logger)

    [ax, bx, qx] = map2twiss(map[0:2,0:2])
    [ay, by, qy] = map2twiss(map[2:4, 2:4])
    [az, bz, qz] = map2twiss(map[4:6,4:6])

    print("Lattice parameters (assuming uncoupled map)", file=logger)
    print("alpha_x: ", ax, " alpha_y: ", ay, file=logger)
    print("beta_x: ", bx, " beta_y: ", by, file=logger)
    print("q_x: ", qx, " q_y: ", qy, file=logger)
    print("q_z: ", qz, " beta_z: ", bz, file=logger)

    (orig_xtune, orig_ytune, orig_cdt) = SIM.Lattice_simulator.calculate_tune_and_cdt(lattice)
    print("Original base tunes, x: ", orig_xtune, " y: ", orig_ytune, file=logger)

    do_adjust_tunes = False
    if opts.xtune or opts.ytune:
        do_adjust_tunes = True
        if opts.xtune:
            target_xtune = opts.xtune
        else:
            target_xtune = orig_xtune
        if opts.ytune:
            target_ytune = opts.ytune
        else:
            target_ytune = orig_ytune

    if do_adjust_tunes:
        print("adjusting tunes, x: ", target_xtune," y: ", target_ytune, file=logger)
        SIM.Lattice_simulator.adjust_tunes(lattice, target_xtune, target_ytune, 1.0e-6)
        (new_xtune, new_ytune, new_cdt) = SIM.Lattice_simulator.calculate_tune_and_cdt(lattice)
        print("Adjusted tunes, x: ", new_xtune, " y: ", new_ytune, file=logger)
        

    chrom = SIM.Lattice_simulator.get_chromaticities(lattice)
    target_xchrom = chrom.horizontal_chromaticity
    target_ychrom = chrom.vertical_chromaticity
    print('initial horizontal chromaticity: ', target_xchrom, file=logger)
    print('initial vertical chromaticity: ', target_ychrom, file=logger)

    adjust_chromaticity = False
    if opts.set_xchrom:
        adjust_chromaticity = True
        target_xchrom = opts.set_xchrom
    if opts.set_ychrom:
        adjust_chromaticity = True
        target_ychrom = opts.set_ychrom

    if adjust_chromaticity:
        print('adjusting chromaticities to: x: ', target_xchrom, ', y: ', target_ychrom, file=logger)
        SIM.Lattice_simulator.adjust_chromaticities(lattice, target_xchrom, target_ychrom, max_steps=20, tolerance=1.0e-2)

    # read back final chromaticity
    chrom = SIM.Lattice_simulator.get_chromaticities(lattice)
    xchrom = chrom.horizontal_chromaticity
    ychrom = chrom.vertical_chromaticity

    alpha_c = chrom.momentum_compaction
    slip_factor = chrom.slip_factor
    print('final horizontal chromaticity: ', xchrom, file=logger)
    print('final vertical chromaticity: ', ychrom, file=logger)
    print("alpha_c: ", alpha_c, ", slip_factor: ", slip_factor, file=logger)


    # save adjusted lattice
    if myrank == 0:
        with open('asymm_lattice.out', 'w') as f:
            print(lattice, file=f)
        with open('asymm_lattice.json', 'w') as f:
            print(lattice.as_json(), file=f)
        lattice.export_madx_file('asymm_lattice.madx', sanitize=True)

    # Get lattice functions and dispersions after lattice
    # adjustments so beam sizes
    # can be calculated for matching

    synergia.simulation.Lattice_simulator.tune_circular_lattice(lattice)
    synergia.simulation.Lattice_simulator.CourantSnyderLatticeFunctions(lattice)
    synergia.simulation.Lattice_simulator.calc_dispersions(lattice)
    lf = lattice.get_elements()[-1].lf
    beta_x = lf.beta.hor
    alpha_x = lf.alpha.hor
    beta_y = lf.beta.ver
    alpha_y = lf.alpha.ver
    psi_x = lf.psi.hor
    psi_y = lf.psi.ver
    disp_x = lf.dispersion.hor
    dprime_x = lf.dPrime.hor

    print('CS lattice functions after adjustments:', file=logger)
    print('beta_x: ', beta_x, file=logger)
    print('alpha_x: ', alpha_x, file=logger)
    print('disp_x: ', disp_x, file=logger)
    print('beta_y: ', beta_y, file=logger)
    print('alpha_y: ', alpha_y, file=logger)

    # create dataframe with the lattice functions
    arclength = [elem.lf.arcLength for elem in lattice.get_elements()]

    beta_xs = [elem.lf.beta.hor for elem in lattice.get_elements()]
    alpha_xs = [elem.lf.beta.hor for elem in lattice.get_elements()]
    psi_xs = [elem.lf.psi.hor for elem in lattice.get_elements()]
    
    beta_ys = [elem.lf.beta.ver for elem in lattice.get_elements()]
    alpha_ys = [elem.lf.beta.ver for elem in lattice.get_elements()]
    psi_ys = [elem.lf.psi.ver for elem in lattice.get_elements()]

    disp_xs = [elem.lf.dispersion.hor for elem in lattice.get_elements()]
    dprime_xs = [elem.lf.dPrime.hor for elem in lattice.get_elements()]


    # stdx = np.sqrt(opts.emitx * beta_x/4 + disp_x**2 * opts.stddpop**2)
    # stdy = np.sqrt(opts.emity * beta_y/4)
    # stddpop = opts.stddpop

    # print("stdx: ", stdx, file=logger)
    # print("stdy: ", stdy, file=logger)
    # print("stdcdt: ", stddpop*bz, file=logger)
    # print("stddpop: ", stddpop, file=logger)

    lfdf = pd.DataFrame({'s': arclength,
                         'beta_x': beta_xs,
                         'alpha_x': alpha_xs,
                         'psi_x': psi_xs,
                         'beta_y' : beta_ys,
                         'alpha_y': alpha_ys,
                         'psi_y': psi_ys,
                         'disp_x': disp_xs,
                         'disp_px': dprime_xs})
    
    # write out lattice functions
    lfdf.to_csv('asymm_lattice_functions.csv')

    ix_lattice = syn2_to_impactx(lattice, init_monitor=False, final_monitor=False)

    for ixelem in ix_lattice:
        ixelem.aperture_x = 3.74 * 0.0254
        ixelem.aperture_y = 1.52 * 0.0254

    if myrank == 0:
        with open('booster_asymm_IX_lattice.py', 'w') as f:
            f.write(elements.KnownElementsList(ix_lattice).to_py())
            
    # Maximum horizontal extent of 3.74" is in the FMAG
    # Maximum vertical extent of 1.52" is in the DMAG element
    # Just set everything to that aperture for simplicity even though
    # we know that other elements (RF cavity for instance) have more
    # more restrictive aperture

if __name__ == "__main__":
    main()

