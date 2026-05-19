#!/usr/bin/env python3
#
# Copyright 2022-2023 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#

import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import pandas as pd
from matplotlib.ticker import MaxNLocator


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


ref_particle = read_time_series("diags/ref_particle.*")

print(len(ref_particle), 'ref particle steps')
n = len(ref_particle)
# steps = nturns*401 + 1
#nturns = (n-1)/401
nturns = (n-1)/1601
print(nturns)
beta = ref_particle.beta[0]
print("beta: ", beta)

t_last = ref_particle["t"].to_numpy()[-1]
print("total length travelled: ", t_last*beta)
print("error in length: ", t_last*beta - nturns*400)
#plt.plot(ref_particle.t)

#plt.show()
