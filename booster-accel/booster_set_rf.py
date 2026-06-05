import numpy as np
import impactx

# set the RF cavity voltage
# phaseR in radians synchronous phase = 0
# total_V is in MV
def set_rf(sim, total_V, freq=freq, phaseR=0, above_transition=False):
    mass = sim.beam.ref.mass_MeV

    rf_cavities = sim.lattice.filter(kind='ShortRF')
    ncav = len(rf_cavities)

    V_per_cav = total_V/ncav

    if above_transition:
        phase = np.pi - phaseR
    else:
        phase = phaseR

    # convert to degrees and subtract 90 to align with ImpactX
    # conventions.
    phase = (180/np.pi) * phase - 90.0

    for elem in rf_cavities:
        elem.V = V_per_cav/mass
        elem.freq = freq
        elem.phase = phase

    return ncav
