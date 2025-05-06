#!/usr/bin/env python3

import sys, os
import glob, re
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import openpmd_api as io

def read_file(file_pattern):
    for filename in glob.glob(file_pattern):
        df = pd.read_csv(filename, delimiter=r"\s+")
        if "step" not in df.columns:
            step = int(re.findall(r"[0-9]+", filename)[0])
            df["step"] = step
        yield df


def read_time_series(file_pattern):
    """Read in all CSV files from each MPI rank (and potentially OpenMP
    thread). Concatenate into one Pandas dataframe.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.concat(
        read_file(file_pattern),
        axis=0,
        ignore_index=True,
    )  # .set_index('id')


def main():
    # ImpactX run reduced-beam-characteraistics (diags)
    rbc = read_time_series('diags/reduced_beam_characteristics.*')

    print('read ImpactX data: ', rbc.shape)

    IX_s = rbc['s']
    IX_sig_x = rbc['sig_x']
    IX_sig_px = rbc['sig_px']
    IX_sig_y = rbc['sig_y']
    IX_sig_py = rbc['sig_py']

    h5 = h5py.File('stepdiag_b000.h5', 'r')
    print('shape(sig_x): ', h5.get('std').shape)
    syn_s = h5.get('s')[()]
    syn_sig_x = h5.get('std')[:, 0]
    syn_sig_px = h5.get('std')[:, 1]
    syn_sig_y = h5.get('std')[:, 2]
    syn_sig_py = h5.get('std')[:, 3]

    print('ImpactX s: ')
    print(IX_s)
    print('Syn s: ')
    print(syn_s)
    print()


    print('ImpactX sig_x: ')
    print(rbc['sig_x'])
    print('Syn sig_x: ')
    print(h5.get('std')[:, 0])
    print()
    
    print('ImpactX sig_px: ')
    print(rbc['sig_px'])
    print('Syn sig_px: ')
    print(h5.get('std')[:, 1])
    print()

    # ImpactX with a monitor as first element has 1 initial step
    # but no final step recorded in reduced_beam_characteristics.

    # Synergia records initial diagnostics but also additional  final
    # diagnostics compared to ImpactX.

    series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)
    iterations = list(series.iterations)
    nsteps = len(iterations)
    print('number steps in series.iterations: ', nsteps)
    # look at particle 0 x, xp
    zref = np.zeros(nsteps)
    p0x = np.zeros(nsteps)
    p0px = np.zeros(nsteps)
    step = 0
    print('iterations: ', iterations)
    for iter in iterations:
        print('iter: ', iter)
        print('len(series.iterations): ', len(series.iterations))
        beam = series.iterations[iter].particles["beam"].to_df()
        zref[step] = series.iterations[iter].particles["beam"].get_attribute('z_ref')
        p0x[step] = beam['position_x'][0]
        p0px[step] = beam['momentum_x'][0]
        del beam

    h5p = h5py.File('tracks_b000.h5', 'r')
    print('shape of track_coords: ', h5p.get('track_coords').shape)
    syn_s = h5p.get('track_s')[()]
    syn_p0x = h5p.get('track_coords')[:, 0, 0]
    syn_p0px = h5p.get('track_coords')[:, 0, 1]
    
    plt.figure()
    stp = np.arange(len(IX_s)-1)
    plt.title('ImpactX vs. Synergia s')
    plt.plot(stp, IX_s[1:], 'o-', label='ImpactX s')
    plt.plot(stp, syn_s[:-1], '^-',  label='Syn s')
    plt.legend(loc='best')
    plt.xlabel('step')

    plt.figure()
    plt.subplot(211)
    plt.plot(IX_sig_x[1:], 'o-', label='ImpactX sig_x')
    plt.plot(syn_sig_x[:-1], 's-', label='Synergia sig_x')
    plt.plot('step')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(IX_sig_px[1:], 'o-', label='ImpactX sig_px')
    plt.plot(syn_sig_px[:-1], 's-', label='Synergia sig_px')
    plt.plot('step')
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(zref, label='ImpactX z')
    plt.plot(syn_s, label='Syn z')
    plt.xlabel('step')
    plt.ylabel('s [m]')
    plt.legend(loc='best')
    
    plt.figure()
    plt.subplot(211)
    plt.plot(p0x, label='ImpactX x')
    plt.plot(syn_p0x, label='Syn x')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.plot(p0px, label='ImpactX px')
    plt.plot(syn_p0px, label='Syn px')
    plt.legend(loc='best')
    plt.xlabel('step')

    print('initial IX sig_x: ', IX_sig_x[0])
    print('initial syn sig_x: ', syn_sig_x[0])

    plt.show()

    series.close()
    h5.close()
    h5p.close()

    pass

if __name__ == "__main__":
    main()
    pass
