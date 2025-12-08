#!/usr/bin/env python3

import sys, os
import numpy as np
import scipy
from scipy.constants import eV
import h5py

from impactx import ImpactX, Config, distribution, elements, amr

# initialize ImpactX first so we can access MPI stuff
sim = ImpactX()

# set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.diagnostics = True  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

mpisize = amr.ParallelDescriptor.NProcs()
myrank = amr.ParallelDescriptor.MyProc()

# open in the input file
h5file = sys.argv[1]
if myrank == 0:
    print('Reading file: ', h5file)

h5 = h5py.File(h5file, 'r')
npart = h5.get('particles').shape[0]
if myrank == 0:
    print(npart, 'particles in file')

# split particles among processes
npart_per_proc = npart//mpisize
npart_extra = npart % mpisize
nparts_by_proc = mpisize * [npart_per_proc]
for i in range(npart_extra):
    nparts_by_proc[i] = nparts_by_proc[i] + 1
partidx = (mpisize+1)*[0]
curidx = 0
for i in range(mpisize):
    partidx[i] = curidx
    curidx = curidx + nparts_by_proc[i]
partidx[-1] = curidx

# are all particles accounted for?
ntot = np.array(nparts_by_proc).sum()
assert curidx == npart
assert ntot == npart

# initialize ImpactX for proper particle parameters
mass = h5.get('mass')[()] # Synergia units are GeV
pz = h5.get('pz')[()]
etot = np.sqrt(pz**2 + mass**2)

# ImpactX units are MeV
kin_energy_MeV = 1.0e3 * (etot - mass)
bunch_charge_C = (6.7e12/81) * scipy.constants.eV # used with space charge

#   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(1.0).set_mass_MeV(mass*1000).set_kin_energy_MeV(kin_energy_MeV)
qm_eev = 1.0 / (mass*1.0e9)  # 1/protom mass  in eV
ref.z = 0

pc = sim.particle_container()

# load up and add my local particles
n_local = nparts_by_proc[myrank]
istart = partidx[myrank]

if not Config.have_gpu:  # initialize using cpu-based PODVectors
    dx_podv = amr.PODVector_real_std()
    dy_podv = amr.PODVector_real_std()
    dt_podv = amr.PODVector_real_std()
    dpx_podv = amr.PODVector_real_std()
    dpy_podv = amr.PODVector_real_std()
    dpt_podv = amr.PODVector_real_std()
else:  # initialize on device using arena/gpu-based PODVectors
    dx_podv = amr.PODVector_real_arena()
    dy_podv = amr.PODVector_real_arena()
    dt_podv = amr.PODVector_real_arena()
    dpx_podv = amr.PODVector_real_arena()
    dpy_podv = amr.PODVector_real_arena()
    dpt_podv = amr.PODVector_real_arena()

for i in range(istart, istart+n_local):
    dx_podv.push_back(h5.get('particles')[i, 0])
for i in range(istart, istart+n_local):
    dy_podv.push_back(h5.get('particles')[i, 2])
for i in range(istart, istart+n_local):
    dt_podv.push_back(h5.get('particles')[i, 4])

for i in range(istart, istart+n_local):
    dpx_podv.push_back(h5.get('particles')[i, 1])
for i in range(istart, istart+n_local):
    dpy_podv.push_back(h5.get('particles')[i, 3])

# dp/p needs to be converted to -dE/p
for i in range(istart, istart+n_local):
    p = (h5.get('particles')[i, 5] + 1) * pz
    dT = -(np.sqrt(p**2 + mass**2) - etot)
    dpt_podv.push_back(dT)

# I don't need the input file any more
h5.close()

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)

# Create lattice with a single monitor element output the distribution.
monitor = elements.BeamMonitor("monitor", backend="h5")

sim.lattice.append(monitor)

sim.track_particles()

sim.finalize()


