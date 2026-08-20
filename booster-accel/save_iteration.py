import sys, os
import numpy as np
import scipy
from scipy.constants import c, eV
import h5py
import openpmd_api as io

from impactx import ImpactX, Config, distribution, elements, amr
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process iteration or index values.")

    # Required positional filename argument
    parser.add_argument("filename", help="Input filename")
    
    parser.add_argument("--iteration", type=int, help="Iteration number")
    parser.add_argument("--index", type=int, help="Index number")

    args = parser.parse_args()

    print(f"Filename provided: {args.filename}")

    if args.iteration is not None:
        print(f"Iteration value provided: {args.iteration}")

    if args.index is not None:
        print(f"Index value provided: {args.index}")

    if args.iteration is None and args.index is None:
        print("No arguments provided. Use --iteration=<number> or --index=<number>")

    # open the input file
    series = io.Series(args.filename, io.Access.read_only)
    iterations = list(series.iterations)
    iter = None
    if args.index is not None:
        iter = iterations[args.index]
        print(f'using index of {args.index}, iteration: {iter}')
    else:
        iter = args.iteration
        if iter not in iterations:
            print(f'iteration {iter} is not in the list of iterations')
        else:
            print(f'using iteration {iter}')

    beam = series.iterations[iter].particles["beam"]
    mass_ref_kg = beam.get_attribute('mass_ref')
    mass_ref_MeV = 1.0e-6 * mass_ref_kg * c**2 /  eV
    print("particle mass (MeV): ", mass_ref_MeV)
    gamma_ref = beam.get_attribute('gamma_ref')
    print('particle energy (MeV): ', gamma_ref * mass_ref_MeV)
    charge_C = beam.get_attribute('charge_C')
    print('charge [C]: ', charge_C)
    s_ref = beam.get_attribute('s_ref')
    t_ref = beam.get_attribute('t_ref')
    x_ref = beam.get_attribute('x_ref')
    px_ref = beam.get_attribute('px_ref')
    y_ref = beam.get_attribute('y_ref')
    py_ref = beam.get_attribute('py_ref')
    z_ref = beam.get_attribute('z_ref')
    pz_ref = beam.get_attribute('pz_ref')

    print('ref s: ', s_ref, ', ref t: ', t_ref)
    df = beam.to_df()
    print(len(df), ' particles in iteration')

    # initialize ImpactX first so we can access MPI stuff
    sim = ImpactX()

    # set numerical parameters and IO control
    sim.particle_shape = 2  # B-spline order
    sim.space_charge = False
    sim.diagnostics = False  # benchmarking
    sim.slice_step_diagnostics = False

    # domain decomposition & space charge mesh
    sim.init_grids()

    # initialize beam based on openPMD file information
    ref = sim.beam.ref
    ref.set_mass_MeV(mass_ref_MeV)
    ref.set_charge_qe(1.0)
    ref.set_kin_energy_MeV(mass_ref_MeV * (gamma_ref - 1))
    qm_eev = 1.0/(mass_ref_MeV * 1.0e6)
    ref.s = s_ref
    ref.t = t_ref
    ref.x = x_ref
    ref.px = px_ref
    ref.y = y_ref
    ref.py= py_ref
    ref.z = z_ref
    ref.pz = pz_ref

    n_local = len(df)

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

    for i in range(n_local):
        dx_podv.push_back(df['position_x'][i])
        dy_podv.push_back(df['position_y'][i])
        dt_podv.push_back(df['position_t'][i])
        dpx_podv.push_back(df['momentum_x'][i])
        dpy_podv.push_back(df['momentum_y'][i])
        dpt_podv.push_back(df['momentum_t'][i])

    pc = sim.beam
    pc.add_n_particles(
        dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, charge_C)

    # create beamline to write out particles
    sim.lattice.clear()

    monitor = elements.BeamMonitor("partsave")
    sim.lattice.append(monitor)

    sim.track_particles()

    sim.finalize()

if __name__ == "__main__":
    main()
