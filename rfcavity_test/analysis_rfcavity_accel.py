import sys, os
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io
import glob
import pandas as pd
from scipy.stats import moment
from scipy.constants import speed_of_light as c, elementary_charge as qe
from pytest import approx
def read_file(file_pattern):
    for filename in glob.glob(file_pattern):
        df = pd.read_csv(filename, delimiter=r"\s+")
        if "step" not in df.columns:
            step = int(re.findall(r"[0-9]+", filename)[0])
            df["step"] = step
        else:
            df = df[df["step"] != "step"]
        df = df.apply(pd.to_numeric)
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

rbc_ref = read_time_series('diags/ref_particle.*')

plt.figure()
plt.title('beta')
plt.plot(rbc_ref['beta'])

plt.figure()
plt.title('gamma')
plt.plot(rbc_ref['gamma'])

plt.show()

series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)

iters = list(series.iterations)

# Make track array
# iteration x particle x coord

niters = len(iters)
s = np.zeros(niters)
gammas = np.zeros(niters)
betas = np.zeros(niters)
betagammas = np.zeros(niters)
mass_ref = series.iterations[iters[0]].particles["beam"].get_attribute('mass_ref')
print("mass_ref: ", mass_ref)

dfs = []
for i, it in enumerate(iters):
    s[i] = series.iterations[it].particles["beam"].get_attribute('s_ref')
    dfs.append(series.iterations[it].particles["beam"].to_df())
    gammas[i] = series.iterations[it].particles["beam"].get_attribute('gamma_ref')
    betas[i] = series.iterations[it].particles["beam"].get_attribute('beta_ref')
    betagammas[i] = series.iterations[it].particles["beam"].get_attribute('beta_gamma_ref')

print('betas')
print(betas)
print()
print('gammas')
print(gammas)
print()
print('beta_gammas')
print(betagammas)

print('dfs[1]')
print(dfs[1])
print()

mvoltage = 1.0/22

momentum_x = np.vstack([ d['momentum_x'] for d in dfs ])
momentum_y = np.vstack([ d['momentum_y'] for d in dfs ])

plt.figure()
for p in range(1,5):
    plt.plot(momentum_x[:, p], lw=3, label=f'px particle {p}')
plt.legend(loc='best')

plt.figure()
for p in range(1,5):
    plt.plot(momentum_y[:, p], lw=3, label=f'py particle {p}')
plt.legend(loc='best')

plt.figure()
plt.title("px particle 2")
plt.plot(momentum_x[:, 2], lw=3, label="px particle 2")
plt.legend(loc='best')

plt.figure()
plt.title("normalized px particle 2")
plt.plot(momentum_x[:, 2]*betagammas, lw=3)

plt.figure()
plt.title("py particle 1")
plt.plot(momentum_y[:, 1], lw=3, label="py particle 1")
plt.legend(loc="best")

plt.figure()
plt.title("normalized py particle 1")
plt.plot(momentum_y[:, 1]*betagammas, lw=3, label="py particle 1")

plt.show()
