import numpy as np
import pandas as pd
from impactx import amr, ImpactX, elements
import openpmd_api as io
from scipy.constants import c, m_p, eV

#opmd_file = "/home/egstern/impactx-apps/pip-ii-painted-distributions/pip-ii-injected-58k-xform-opmd.h5"
#opmd_file = '/home/egstern/impactx-apps/booster-simple/run100y.00/diags/openPMD/monitor.h5'
#opmd_file = "/home/egstern/impactx-apps/pip-ii-painted-distributions/pip-ii-injected-5.8k-xform-opmd.h5"
opmd_file = './diags_part/openPMD/monitor.h5'

# Initialize ImpactX and grids
sim = ImpactX()

sim.init_grids()

mpi_rank = amr.ParallelDescriptor.MyProc()
mpi_size = amr.ParallelDescriptor.NProcs()

print(f"Hi! This is MPI rank {mpi_rank}/{mpi_size}")

# get beam information from openPMD file
series = io.Series(opmd_file, io.Access.read_only)

print(f"MPI rank ", mpi_rank, "opening file ", opmd_file)

iters = list(series.iterations)

first = iters[0]
firstbeam = series.iterations[first].particles["beam"]

charge_ref = firstbeam.get_attribute('charge_ref')
mass_ref = firstbeam.get_attribute('mass_ref')
beta_gamma_ref = firstbeam.get_attribute('beta_gamma_ref')
gamma_ref = firstbeam.get_attribute('gamma_ref')
#print(help(firstbeam.to_df))
file_df = firstbeam.to_df()
print("file_df columns: ", file_df.columns)
npart_in_file = len(file_df)
if mpi_rank == 0:
    print('first ten file IDs: ', file_df.shape)
    print(file_df.loc[:, 'id'][:10])
    f = open('opmd_file_ids.lis', 'w')
    print(file_df.loc[:, 'id'][:10], file=f)
    f.close()

if mpi_rank == 0:
    print('mass_ref: ', firstbeam.get_attribute('mass_ref'))
    print('num particles in file: ', npart_in_file)

series.close()

m_mev = 1.0e-6 * mass_ref * c**2/charge_ref
if mpi_rank == 0:
    print('m_mev: ', m_mev)
    print('gamma_ref: ', gamma_ref)

ref = sim.beam.ref
ref.set_mass_MeV(m_mev)
ref.set_kin_energy_MeV((gamma_ref-1)*m_mev)
ref.set_charge_qe(1)

source = elements.Source("openPMD", opmd_file)
sim.lattice.clear()
sim.lattice.append(source)

sim.track_particles()

print(f'number_of_particles on rank {mpi_rank}: {sim.beam.number_of_particles(only_local=True)}')
print(f'total_number_of_particles on rank {mpi_rank}: {sim.beam.total_number_of_particles()}', flush=True)
local_df = sim.beam.to_df(local=True)
print(f'size of sim.beam dataframe on rank {mpi_rank}: {len(local_df)}', flush=True)
if mpi_rank == 0:
    print('local_df columns: ', local_df.columns, flush=True)

with open('pc_df_ids_{}.lis'.format(mpi_rank), 'w') as f:
    #print(local_df.loc[:, 'idcpu'][:10], file=f, flush=True)
    print(f'MPI rank {mpi_rank} x_mean: {local_df['position_x'].mean()}', flush=True, file=f)
    print(f'MPI rank {mpi_rank} x_std: {local_df['position_x'].std()}', flush=True, file=f)
    print(f'MPI rank {mpi_rank} x_min/x_max: {local_df['position_x'].min()}/{local_df['position_x'].max()}', flush=True, file=f)

print(f'MPI rank {mpi_rank} x_mean: {local_df['position_x'].mean()}', flush=True)
print(f'MPI rank {mpi_rank} x_std: {local_df['position_x'].std()}', flush=True)
print(f'MPI rank {mpi_rank} x_min/x_max: {local_df['position_x'].min()}/{local_df['position_x'].max()}', flush=True)

sim.finalize()

