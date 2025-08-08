#!/usr/bin/env python

# analyze multipole run
import sys, os
import numpy as np
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

runs = ["k1", "k1s", "k2", "k2s", "k3", "k3s", "k4", "k4s", "k5", "k5s",
        "k6", "k6s"]

# run is a string like "k3s"
def analyze_run(run):
    series = io.Series(f"diags/openPMD/{run}.h5", io.Access.read_only)
    npdat = np.load(f"mpole_{run}.npy")

    print("reading data from ", run)

    iterations = list(series.iterations)
    beam = series.iterations[iterations[-1]].particles["beam"].to_df()

    #print(beam)
    #print(beam.shape)
    npart = beam.shape[0]
    mom_x = beam['momentum_x']
    mom_y = beam['momentum_y']
    
    for i in range(npart):
        bad = 0
        if mom_x[i] != approx(npdat[i, 1], rel=1.0e-12, abs=1.0e-15):
            bad += 1
        if mom_y[i] != approx(npdat[i, 3], rel=1.0e-12, abs=1.0e-15):
            bad += 2
        if bad != 0:
            print(f'Failure {run} particle {i}')
            if bad & 0x01 != 0:
                print(f'    mom_x: {mom_x[i]} should be {npdat[i, 1]}')
            if bad & 0x02 != 0:
                print(f'    mom_y: {mom_y[i]} should be {npdat[i, 3]}')
            print(f'particle at position {pos_x[i]}, {pos_y[i]}')
            print()
    # for i in range(npart):
    #     print(npdat[i, 1], mom_x[i])
    # print()
    # for i in range(npart):
    #     print(npdat[i, 3], mom_y[i])
    # series.close()

    return

def main():
    for r in runs:
        analyze_run(r)
    pass

if __name__ == "__main__":
    main()
    pass
