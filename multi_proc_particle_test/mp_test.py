import sys, os
#import mpi4py.MPI as MPI
import numpy as np
from scipy import constants
#import synergia

from impactx import ImpactX, distribution, elements, amr

#import impactx
#import amrex.space3d as amr

sim = ImpactX()

# need to initialize sim in order to see MPI
sim.init_grids()

#print('dir(sim): ', dir(sim))

#print('dir(amr): ', dir(amr))

      

#print('impactx.amr')
#print(dir(impactx.amr))

size = amr.ParallelDescriptor.NProcs()
myrank = amr.ParallelDescriptor.MyProc()

print(f'This is rank {myrank} out of {size} procs')

sim.finalize()

sys.exit(0)

