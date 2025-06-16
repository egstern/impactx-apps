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

def load_rbc(file):
    df = pd.read_csv(file, delimiter=r"\s+")
    return df

def main():
    dfs = []
    for f in sys.argv[1:]:
        print('reading file ', f)
        df = load_rbc(f)
        dfs.append(df)
        pass

    print('read ', len(dfs), ' RBCs')

    if len(dfs) <= 1:
        return

    q = dfs[0].columns
    print('Quantities to check: ', q)
    
    for i in q[1:]:
        print('checking quantity: ', i)
        for j in range(1, len(dfs)):
            print('checking 0 vs. ', j, '... ', end='')
            diff = False
            if ((dfs[0][i] - dfs[j][i]) != 0.0).any():
                print('difference in ', j, ' for variable ', i)
                diff = True
                break
            if diff:
                print('difference found')
            else:
                print('OK')
    return


if __name__ == "__main__":
    main()
    pass
