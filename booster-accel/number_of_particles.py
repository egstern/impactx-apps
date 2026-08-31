import sys
import numpy as np

from scipy.constants import c, eV, m_p, pi
from impactx import ImpactX, distribution, elements, twiss, synmadx
import amrex.space3d as amr

from scipy.constants import c

def main(inp_source):

    sim = ImpactX()

    # set numerical parameters and IO control
    sim.space_charge = False
    # sim.diagnostics = False  # benchmarking
    # set slice step diagnostics
    sim.slice_step_diagnostics = True
    
    # domain decomposition & space charge mesh
    sim.init_grids()
    
    sim.lattice.clear()
    source = elements.Source(distribution="openPMD", openpmd_path=inp_source)
    sim.lattice.append(source)

    sim.track_particles()

    print("Number of particles from beam: ", sim.beam.total_number_of_particles())
    df = sim.beam.to_df(local=False)
    print("total number of particles from df: ", len(df))

    sim.finalize()
    return

if __name__ == "__main__":
    inp_source = sys.argv[1]
    print("Input source file: ", inp_source)

    main(inp_source)
