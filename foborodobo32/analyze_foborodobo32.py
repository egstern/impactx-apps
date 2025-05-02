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
    IX_sig_x = rbc['sig_x']
    IX_sig_y = rbc['sig_y']

    h5 = h5py.File('stepdiag_b000.h5', 'r')
    print('shape(sig_x): ', h5.get('std').shape)
    syn_sig_x = h5.get('std')[:, 0]
    syn_sig_y = h5.get('std')[:, 2]

    plt.plot(IX_sig_x, 'o-', label='ImpactX sig_x')
    plt.plot(syn_sig_x, 's-', label='Synergia sig_x')
    plt.plot('step')
    plt.legend(loc='best')

    print('initial IX sig_x: ', IX_sig_x[0])
    print('initial syn sig_x: ', syn_sig_x[0])

    plt.show()


    pass

if __name__ == "__main__":
    main()
    pass
