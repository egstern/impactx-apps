#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpmd_api as io
from scipy.stats import moment
from pytest import approx

def plot_data():

    series = io.Series("diags/openPMD/monitor.h5", io.Access.read_only)
    iterations = list(series.iterations)

    start = series.iterations[iterations[0]]
    finish = series.iterations[iterations[-1]]

    start_beam = start.particles["beam"].to_df()
    finish_beam = finish.particles["beam"].to_df()

    print('start_beam-x')
    print(start_beam['position_x'][:25])

    print()
    print('start_beam-y')
    print(start_beam['position_y'][:25])

    # All start momenta better be 0
    is_px_0 = (start_beam['momentum_x'] == 0.0).any()
    is_py_0 = (start_beam['momentum_y'] == 0.0).any()
    is_pt_0 = (start_beam['momentum_t'] == 0.0).any()
    print('is_px_0: ', is_px_0)
    print('is_py_0: ', is_py_0)
    print('is_pt_0: ', is_pt_0)

    plt.figure()
    plt.title('start beam transverse')
    plt.plot(start_beam['position_x'], start_beam['position_y'], '.')
    plt.xlabel('x')
    plt.ylabel('y')

    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(start_beam['position_t'], start_beam['position_x'], '.', label='x')
    axs[0].legend(loc='best')
    axs[1].plot(start_beam['position_t'], start_beam['position_y'], '.', label='y')
    axs[1].legend(loc='best')
    axs[1].set_xlabel('t')

    plt.show()
    return

def main():
    plot_data()
    return

if __name__ == "__main__":
    main()
    pass
